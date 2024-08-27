import os
import gc
import json
import time
import torch
import random
import torch.nn as nn
from main import getmisc, search101_custom
from measures import get_grad_score_by_measure
from nasbench.api import NASBench as API101, ModelSpec


class StaticArgs:
    searchspace = "101"
    dataset = "cifar10"
    data_path = "~/dataset/"
    cutout = 0
    batchsize = 128
    num_worker = 1
    metric = "grad"
    startnetid = 0
    manualSeed = 0
    n_runs = 1
    sampling = "S"
    regularize = "oldest"


def ge_nasbench_101(api_101: API101, metric, args=None, C=200, P=10, S=5, iteration=0):
    print(metric, iteration, "start")
    execution_start = time.time()
    if args is None:
        args = StaticArgs()

    def mutate(idx):
        to_do = ["change_op"] * 4
        ops_available = ['conv1x1-bn-relu', 'conv3x3-bn-relu', "maxpool3x3"]
        fixed, _ =  api_101.get_metrics_from_hash(idx)

        ops = fixed['module_operations']
        adjacency = fixed['module_adjacency'].tolist()
        _len = len(ops)

        if _len > 2:
            to_do.append("shrink")

        if _len < 7:
            to_do.append("enlarge")

        to_do = random.choice(to_do)

        if to_do == "shrink":
            p = random.randint(1, _len - 2)
            ops = ops[:p] + ops[p+1:]
            adjacency = adjacency[:p] + adjacency[p+1:]
            for i in range(len(adjacency)):
                adjacency[i] = adjacency[i][:p-1] + adjacency[i][p:]
        elif to_do == "enlarge":
            new_op = random.choice(ops_available)
            ops = ops[:-1] + [new_op] + ops[-1:]
            for i in range(len(adjacency)):
                adjacency[i] = adjacency[i] + [0]
            adjacency[-1][-1] = 1
            adjacency.append([0] * (_len + 1))
        elif to_do == "change_op":
            p = random.randint(1, _len - 2)
            new_ops = [op for op in ops_available]
            random.shuffle(new_ops)
            if new_ops[0] != ops[p]:
                ops[p] = new_ops[0]
            else:
                ops[p] = new_ops[1]

        # for ad in adjacency: print(ad)
        new_spec = ModelSpec(adjacency, ops)
        if api_101.is_valid(new_spec):
            new_idx = api_101._hash_spec(new_spec)
            if new_idx in api_101.computed_statistics:
                return new_idx

        return -1

    def mutation(idx, history, generation):
        max_tries_history = 15
        max_tries_generation = 10
        max_tries = max(max_tries_history, max_tries_generation) + 1

        for n_try in range(max_tries):
            try:
                new_net = mutate(idx)
            except:
                new_net = -1

            if new_net == -1:
                continue

            if max_tries_history < n_try and new_net in history:
                continue

            if max_tries_generation < n_try and new_net in generation:
                continue

            return new_net

        return idx


    args = StaticArgs()
    imgsize, ce_loss, trainloader, testloader = getmisc(args)

    def cast_candidate(idx, s, s_time, info, epochs=12):
        return {
            "net_id": idx,
            "score": s,
            "epochs": epochs,
            "trained": False,
            "score_time": s_time,
            "val_acc": info[2],
            "test_acc": info[1],
            "train_acc": info[0],
            "train_time": info[3],
        }

    def cast_individual(candidate):
        r = {k: v for k, v in candidate.items()}
        r["trained"] = True
        return r

    for i, batch in enumerate(trainloader):
        data, label = batch[0], batch[1]
        data, label = data.cuda(), label.cuda()
        if iteration <= i:
            break

    init_population = []
    generations = []

    rand_init = random.sample(list(api_101.hash_iterator()), C)

    score_time = 0
    train_time_cost = 0

    while len(init_population) < C:

        idx = rand_init[len(init_population)]
        # print("idx", idx)
        network, info = search101_custom(api_101, idx)
        # network.cuda()
        # s_timestamp = time.time()
        # s = get_grad_score_by_measure(
        #     metric, network, data, label, ce_loss, split_data=1, device="cuda"
        # )
        # s_time = time.time() - s_timestamp
        s = info[1]
        s_time = info[3]
        score_time = score_time + s_time
        candidate = cast_candidate(idx, s, s_time, info)
        init_population.append(candidate)

        del network
        torch.cuda.empty_cache()
        gc.collect()


    init_population = sorted(init_population, key=lambda x: x["score"], reverse=True)
    print(metric, iteration, "initial_population_done")

    history = []
    population = []

    for ind in reversed(init_population[:P]):
        ind_value = cast_individual(ind)
        history.append(ind_value)
        population.append(ind_value)
        train_time_cost += ind_value["train_time"]

    C_AUX = C  # cyles
    while C_AUX >= 0:
        C_AUX -= 1
        sample = []
        if args.sampling == "S":  # sample S candidates
            sample = random.sample(population, S)
        elif args.sampling in ["lowest", "highest"]:
            sample = sorted(
                population,
                key=lambda i: i["score"],
                reverse=args.sampling == "highest",
            )[:S]

        parent = max(sample, key=lambda i: i["test_acc"])

        P_AUX = P
        generation = []
        while len(generation) <= P_AUX:

            h_idx = [x["net_id"] for x in history]
            g_idx = [x["net_id"] for x in generation]
            new_idx = mutation(parent["net_id"], h_idx, g_idx)
            # network, info = search101_custom(
            #     api_101, new_idx
            # )
            # network.cuda()
            # s_timestamp = time.time()
            # s = get_grad_score_by_measure(
            #     metric, network, data, label, ce_loss, split_data=1, device="cuda"
            # )
            s = info[1]
            s_time = info[3]
            s_time = time.time() - s_timestamp
            score_time = score_time + s_time
            candidate = cast_candidate(new_idx, s, s_time, info)
            generation.append(candidate)

            # del network
            torch.cuda.empty_cache()
            gc.collect()

        chosen_ind = max(generation, key=lambda i: i["score"])  # get highest score

        ind_value = cast_individual(chosen_ind)
        generations.append(
            {
                "parent": parent,
                "sample": sample,
                "figthers": generation,
                "chosen": ind_value,
            }
        )
        history.append(ind_value)  # add ind to history
        population.append(ind_value)  # add ind to current pop
        train_time_cost += ind_value["train_time"]

        if args.regularize == "oldest":
            indv = population[0]  # oldest
        elif args.regularize == "lowest":  # remove lowest scoring
            indv = min(population, key=lambda i: i["test_acc"])  # min value
        elif args.regularize == "highest":  # remove highest scoring
            indv = max(population, key=lambda i: i["test_acc"])  # min value
        population.pop(population.index(indv))

    top_scoring_arch = max(history, key=lambda i: i["test_acc"])
    _, info = search101_custom(
        api_101, top_scoring_arch["net_id"], 108
    )
    top_arch = cast_individual(
        cast_candidate(
            top_scoring_arch["net_id"],
            top_scoring_arch["score"],
            0,
            info,
            108,
        )
    )
    execution_time_cost = time.time() - execution_start
    print(metric, iteration, "executed")
    res = {
        "it": iteration,
        "metric": metric,
        "top_arch": top_arch,
        "init_population": init_population,
        "generations": generations,
        "score_time": score_time,
        "train_time": train_time_cost + top_arch["train_time"],
        "execution_time": execution_time_cost,
    }
    print("res", res)
    return res


def instance(api_101: API101, metric, args=None, C=200, P=10, S=5, iteration=0):
    filename = "/content/drive/MyDrive/Exp003_workspace/%s_%02d" % (metric, iteration)

    res = ge_nasbench_101(api_101, metric, args, C, P, S, iteration)

    with open(filename, "w") as f:
        json.dump(res, f)
