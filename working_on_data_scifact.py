import pandas as pd

# df = pd.read_csv(r"D:\ir-project\data\nfcorpus\qrels\test.tsv",sep="\t")
df_scifact = pd.read_csv(r"D:\ir-project\data\scifact\qrels\test.tsv",sep="\t")
print(df_scifact["query-id"].nunique())

queries = pd.read_json(r"D:\ir-project\data\scifact\queries.jsonl",lines=True)

print(queries["_id"].count())
print(queries["_id"].nunique())

df_nfc = pd.read_csv(r"D:\ir-project\data\nfcorpus\qrels\test.tsv",sep="\t")
print(df_nfc["query-id"].nunique())
nfcorpus = pd.read_json(r"D:\ir-project\data\nfcorpus\corpus.jsonl",lines=True)
queries_nfc = pd.read_json(r"D:\ir-project\data\nfcorpus\queries.jsonl",lines=True)
print(nfcorpus["_id"].nunique())
print(queries_nfc["_id"].count())
print(queries_nfc["_id"].nunique())