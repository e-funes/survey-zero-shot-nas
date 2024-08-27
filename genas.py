import os
import json
import time
import random
from main import getmisc, search201_custom
from measures import get_grad_score_by_measure
from nas_201_api import NASBench201API as API201


class StaticArgs:
    searchspace = "201"
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


def ge_nasbench_201(api_201: API201, metric, args=None, C=200, P=10, S=5, iteration=0):
    print(metric, iteration, "start")
    execution_start = time.time()
    if args is None:
        args = StaticArgs()

    def mutation(idx, history, generation):
        max_tries_history = 15
        max_tries_generation = 10
        max_tries = max(max_tries_history, max_tries_generation) + 1
        ops = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]

        def to_list(tuples):
            def t2l(t):
                return [t[0], t[1]]

            [t1, t2, t3] = tuples
            return [
                t2l(t1[0]),
                t2l(t2[0]),
                t2l(t2[1]),
                t2l(t3[0]),
                t2l(t3[1]),
                t2l(t3[2]),
            ]

        def to_str(_list):
            l0 = ["%s~%s" % (s[0], s[1]) for s in _list]
            l0 = [
                "|" + "|".join(l0[0:1]) + "|",
                "|" + "|".join(l0[1:3]) + "|",
                "|" + "|".join(l0[3:6]) + "|",
            ]
            return "+".join(l0)

        for n_try in range(max_tries):
            net_conf = api_201.get_net_config(idx, args.dataset)
            _list = to_list(api_201.str2lists(net_conf["arch_str"]))
            _idx = random.randint(0, len(_list) - 1)
            [_pick, _pick2] = random.sample(ops, 2)
            _list[_idx][0] = _pick if _pick != _list[_idx][0] else _pick2
            _list = to_str(_list)
            new_net = api_201.query_index_by_arch(_list)

            if new_net == -1:
                continue

            if max_tries_history < n_try and new_net in history:
                continue

            if max_tries_generation < n_try and new_net in generation:
                continue

            return new_net

        return idx

    args = StaticArgs()
    # imgsize, ce_loss, trainloader, testloader = getmisc(args)

    def cast_candidate(idx, s, s_time, train_info, test_info, epochs=12):
        return {
            "net_id": idx,
            "score": s,
            "score_time": s_time,
            "epochs": epochs,
            "trained": False,
            "test_acc": test_info[1],
            "test_loss": test_info[0],
            "test_time": test_info[2],
            "train_acc": train_info[1],
            "train_loss": train_info[0],
            "train_time": train_info[2],
        }

    def cast_individual(candidate):
        r = {k: v for k, v in candidate.items()}
        r["trained"] = True
        return r

    # for i, batch in enumerate(trainloader):
    #     data, label = batch[0], batch[1]
    #     data, label = data.cuda(), label.cuda()
    #     if iteration <= i:
    #         break

    init_population = []
    generations = []

    rand_init = random.sample(range(len(api_201.meta_archs)), C)

    score_time = 0
    train_time_cost = 0

    while len(init_population) < C:
        idx = rand_init[len(init_population)]
        network, train_info, test_info = search201_custom(api_201, idx, args.dataset)
        # network.cuda()
        # s_timestamp = time.time()
        # s = get_grad_score_by_measure(
        #     metric, network, data, label, ce_loss, split_data=1, device="cuda"
        # )
        # s_time = time.time() - s_timestamp
        s = train_info[1]
        s_time = train_info[2]
        score_time = score_time + s_time
        candidate = cast_candidate(idx, s, s_time, train_info, test_info)
        init_population.append(candidate)

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
            network, train_info, test_info = search201_custom(
                api_201, new_idx, args.dataset
            )
            # network.cuda()
            # s_timestamp = time.time()
            # s = get_grad_score_by_measure(
            #     metric, network, data, label, ce_loss, split_data=1, device="cuda"
            # )
            # s_time = time.time() - s_timestamp
            s = train_info[1]
            s_time = train_info[2]
            score_time = score_time + s_time
            candidate = cast_candidate(new_idx, s, s_time, train_info, test_info)
            generation.append(candidate)

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
    _, top_train_info, top_test_info = search201_custom(
        api_201, top_scoring_arch["net_id"], args.dataset, "200"
    )
    top_arch = cast_individual(
        cast_candidate(
            top_scoring_arch["net_id"],
            top_scoring_arch["score"],
            0,
            top_train_info,
            top_test_info,
            200,
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


def instance(api_201: API201, metric, args=None, C=200, P=10, S=5, iteration=0):
    filename = "/content/drive/MyDrive/Exp003_workspace/%s_%02d" % (metric, iteration)

    res = ge_nasbench_201(api_201, metric, args, C, P, S, iteration)

    with open(filename, "w") as f:
        json.dump(res, f)
