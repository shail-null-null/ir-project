import math

def precision_at_k(ranked,rels,k):
    ans = 0.0
    for i in ranked[:k]:
        if rels.get(i, 0) >= 1:
            ans += 1
    ans = ans/k
    return ans

def average_precision(ranked,rels):
    ans = 0.0
    temp = 0.0
    r_avg = 0.0
    count = 0
    for i in rels:
        if rels[i]>0:
            count += 1
    if count==0:
        return 0
    for i in range(len(ranked)):
        if rels.get(ranked[i], 0) >= 1:
            temp += 1
            r_avg = temp/(i+1)
            ans += r_avg
    ans = ans/count
    return ans

def ndcg_at_k(ranked,rels,k):
    ideal = 0.0
    sorted_rels = dict(sorted(rels.items(),key=lambda item : item[1],reverse=True))
    i = 0
    for key in sorted_rels:
        if i >= k:
            break
        ideal += rels[key]/(math.log(i+2,2))
        i += 1
    score = 0.0
    index = 0
    for j in ranked[:k]:
        if j in rels.keys():
            score += rels[j]/(math.log(index+2,2))
        index += 1
    if ideal == 0:             
        return 0.0
    ndcg = score/ideal
    return ndcg

def evaluate(run,qrels,k):
    scores = {}
    for key in qrels:
        l = []
        if key in run:
            sorted_run = dict(sorted(run[key].items(),key=lambda item:item[1],reverse=True))
            for z in sorted_run:
                l.append(z)
            patk = precision_at_k(l,qrels[key],k)
            ap = average_precision(l,qrels[key])
            ndcg = ndcg_at_k(l,qrels[key],k)
            scores[key] = {"P@3":patk,"AP": ap,"NDCG@3":ndcg}
        if key not in run:
            scores[key] = {f"P@3":0,"AP": 0,f"NDCG@3":0}
    return scores


if __name__  == "__main__":
    ranked_q1 = ["d3", "d2", "d1", "d4", "d6"];  rels_q1 = {"d1": 2, "d3": 1, "d6": 1}
    ranked_q2 = ["d7", "d5", "d9"];              rels_q2 = {"d5": 1, "d8": 1}
    run   = {"q1": {"d3": 5.0, "d2": 4.0, "d1": 3.0, "d4": 2.0, "d6": 1.0},
            "q2": {"d7": 3.0, "d5": 2.0, "d9": 1.0}}
    qrels = {"q1": {"d1": 2, "d3": 1, "d6": 1},
            "q2": {"d5": 1, "d8": 1},
            "q3": {"d2": 1}}
    # patk_q1 = precision_at_k(ranked_q1,rels_q1,3)
    # patk_q2 = precision_at_k(ranked_q2,rels_q2,3)
    # print(round(patk_q1,3))
    # print(round(patk_q2,3))
    # ap_q1 = average_precision(ranked_q1,rels_q1)
    # ap_q2 = average_precision(ranked_q2,rels_q2)
    # print(round(ap_q1,3))
    # print(round(ap_q2,3))
    # ndcg_at_3_q1 = ndcg_at_k(ranked_q1,rels_q1,3)
    # ndcg_at_5_q1 = ndcg_at_k(ranked_q1,rels_q1,5)
    # ndcg_at_3_q2 = ndcg_at_k(ranked_q2,rels_q2,2)
    # print(round(ndcg_at_3_q1,3))
    # print(round(ndcg_at_5_q1,3))
    # print(round(ndcg_at_3_q2,3))
    scores = evaluate(run,qrels,3)
    print(scores)