import pandas as pd
import json

def Run(filename):
    # id 要做 題組內外的區分
    data_path = "/home/csi/project_NLP/data/"
    filepath = data_path + filename
    collect_df = pd.read_csv(filepath)
    in_user = []
    out_user = []
    for user in collect_df["id"].unique():
        user_df = collect_df[collect_df["id"]==user]
        if len(user_df[user_df["動作"]=="打分"]) > 0:
            if len(user_df[~user_df["意圖"].isna()]) >= len(user_df[user_df["意圖"].isna()]):
                in_user.append(user)
            else:
                out_user.append(user)
    usr_df = collect_df[collect_df["id"].isin(in_user)]
    test_df = usr_df.copy()
    table1 = {}
    statified_df = test_df[test_df["動作"]=="打分"]
    yes = 0; soso = 0; no = 0
    for user in statified_df["id"].unique():
        summary_df = statified_df[statified_df["id"]==user].reset_index().sort_values(by=["時間"], ascending=False)
        if summary_df.loc[:, "內容"][0] == "滿意":
            yes += 1
        elif summary_df.loc[:, "內容"][0] == "不滿意":
            no += 1
        elif summary_df.loc[:, "內容"][0] == "普通":
            soso += 1
    table1["滿意"]=yes
    table1["不滿意"]=no
    table1["普通"]=soso

    table2 = {}
    sel_df = collect_df[collect_df["動作"]!="打分"]
    table2["題組內"] = len(sel_df[~sel_df["意圖"].isna()])
    table2["題組外"] = len(sel_df[sel_df["意圖"].isna()])

    table = {}
    table["題組數"] = table2
    table["滿意度"] = table1
    return json.dumps(table, ensure_ascii=False)

