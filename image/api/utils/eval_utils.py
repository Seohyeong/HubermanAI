import json
import pandas as pd
from tqdm import tqdm
import mlflow
    

def get_rr(gt_doc_id: str, pred_doc_ids: list[str]) -> float:
    rr= 0
    try:
        rr = 1 / (pred_doc_ids.index(gt_doc_id) + 1)
    except ValueError:
        rr = 0
    return rr

def get_mrr(data, rag_chain, k) -> float:
    mrr = 0
    for item in tqdm(data, total=len(data)):
        output = rag_chain.retrieve(item["question"], k)
        retrieved_docs = output.docs
        gt_doc_id = item["doc_id"]
        pred_doc_ids = []
        for doc in retrieved_docs:
            doc_id = doc.video_id + "_" + doc.segment_idx
            pred_doc_ids.append(doc_id)
        rr = get_rr(gt_doc_id, pred_doc_ids)
        mrr += rr
    mrr = mrr / len(data)
    return mrr

def get_recall(data, rag_chain, k) -> float:
    recall = 0
    for item in tqdm(data, total=len(data)):
        output = rag_chain.retrieve(item["question"], k)
        retrieved_docs = output.docs
        gt_doc_id = item["doc_id"]
        pred_doc_ids = []
        for doc in retrieved_docs:
            doc_id = doc.video_id + "_" + doc.segment_idx
            pred_doc_ids.append(doc_id)
        if gt_doc_id in pred_doc_ids:
            recall += 1
    recall = recall / len(data)
    return recall

def get_score_dist(data, rag_chain, k) -> list[float]:
    scores = []
    for item in tqdm(data, total=len(data)):
        output = rag_chain.retrieve(item["question"], k)
        retrieved_docs = output.docs
        pred_doc_ids = []
        for doc in retrieved_docs:
            scores.append(doc.score)
            doc_id = doc.video_id + "_" + doc.segment_idx
            pred_doc_ids.append(doc_id)
    return scores
    
def get_confusion_matrix(data: list, threshold: float, relevant: bool) -> tuple[float, float]:
    n = len(data)
    # True Positiv Rate: Percentage of relevant queries that correctly retrieved documents above threshold
    # False Negative Rate: Percentage of relevant queries that correctly resulted in no documents above the threshold
    # True Negative Rate: Percentage of irrelevant queries that correctly resulted in no documents above the threshold
    # False Positive Rate: Percentage of irrelevant queries that incorrectly retrieved documents above threshold
    if relevant:
        tp, fn = 0, 0
        for score in data:
            if score >= threshold:
                tp += 1
            else:
                fn += 1
        return (tp/n, fn/n)
    else:
        tn, fp = 0, 0
        for score in data:
            if score <= threshold:
                tn += 1
            else:
                fp += 1
        return (tn/n, fp/n)

def test_retriever(rag_chain, k, threshold):
    """ checking two things
    - score distribution for relevant questions (syn_test_data, qna_test_data)
    - recall, mrr (syn_test_data)
    """
    run_name = f"{rag_chain.embedding_function}_retrieval_eval"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_name", rag_chain.llm_model_name)
        mlflow.log_param("k", k)
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("embedding_model", rag_chain.config.embedding_model)
        mlflow.log_param("search_type", rag_chain.config.search_type)
        
        # for IR recall, mrr
        with open(rag_chain.config.syn_test_data_path, "r", encoding = "utf-8") as f:
            syn_data = [json.loads(line) for line in f]
        # for score dist
        with open(rag_chain.config.qna_test_data_path, "r", encoding = "utf-8") as f:
            qna_data = [json.loads(line) for line in f]
        with open(rag_chain.config.relevant_qs_path, "r", encoding = "utf-8") as f:
            relevant_questions = [json.loads(line) for line in f]
        with open(rag_chain.config.irrelevant_qs_path, "r", encoding = "utf-8") as f:
            irrelevant_questions = [json.loads(line) for line in f]
            
        syn_data_mrr = get_mrr(syn_data, rag_chain, k)
        syn_data_recall = get_recall(syn_data, rag_chain, k)
        qna_data_scores = get_score_dist(qna_data, rag_chain, k)
        rel_data_scores = get_score_dist(relevant_questions, rag_chain, k)
        irrel_data_scores = get_score_dist(irrelevant_questions, rag_chain, k)
        
        tp, fn = get_confusion_matrix(rel_data_scores + qna_data_scores, threshold=threshold, relevant=True)
        tn, fp = get_confusion_matrix(irrel_data_scores, threshold=threshold, relevant=False)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        # Log metrics to MLflow
        mlflow.log_metric(f"mrr_{k}", syn_data_mrr)
        mlflow.log_metric(f"recall_{k}", syn_data_recall)
        mlflow.log_metric("true_positives", tp)
        mlflow.log_metric("true_negatives", tn)
        mlflow.log_metric("false_positives", fp)
        mlflow.log_metric("false_negatives", fn)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)  # Named differently to avoid conflict
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("accuracy", accuracy)
        
        # Log score distribution statistics
        qna_stats = pd.Series(qna_data_scores).describe()
        rel_stats = pd.Series(rel_data_scores).describe()
        irrel_stats = pd.Series(irrel_data_scores).describe()
        
        for stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
            if '%' in stat:
                mlflow_stat = stat.strip('%')
            else:
                mlflow_stat = stat
            mlflow.log_metric(f"qna_scores_{mlflow_stat}", qna_stats[stat])
            mlflow.log_metric(f"relevant_scores_{mlflow_stat}", rel_stats[stat])
            mlflow.log_metric(f"irrelevant_scores_{mlflow_stat}", irrel_stats[stat])
        
        print(
            f"RECALL: {syn_data_mrr}\n"
            f"MRR: {syn_data_recall}\n\n"
            f"CONFUSION MATRIX({threshold}, {k})\n"
            f"TP: {tp}, TN: {tn}\n"
            f"FP: {fp}, FN: {fn}\n\n"
            f"SUMMARY(QNA):\n{pd.Series(qna_data_scores).describe()}\n"
            f"SUMMARY(RELATED):\n{pd.Series(rel_data_scores).describe()}\n"
            f"SUMMARY(UNRELATED):\n{pd.Series(irrel_data_scores).describe()}"
        )