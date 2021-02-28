
"""
モデリングおよび解釈パートのコード資産（宮田パート）
ラインナップは以下
    - LightGBMでのfit関数（二値分類）
    - LightGBMでのpredict関数（二値分類）
    - valid　と　test　の予測結果の分布状況を可視化（確率密度で）
    - feature_importanceの可視化
    - ROC曲線の可視化
    - テーブル同士での各カラムごとの重複度合いをベン図で可視化
    - 作成画像を.png形式で所定場所に保存する
"""

# Libraries
import os
import numpy as np
import pandas as pd
import pickle

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from matplotlib import pyplot
import japanize_matplotlib
# %matplotlib inline

import lightgbm as lgbm
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import log_loss, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import shap


# LightGBMでのfit関数（二値分類）
def fit_lgbm(feat_train_df, target_df, cv=None, params: dict=None, verbose=500, do_shap=False):
    """
    LightGBMで学習させたmodelを作成する
    学習後、AUCスコアもprint出力される
    Args:
        feat_train_df: 説明変数のDataFrame
        target_df: 目的変数のDataFrame
        cv: クロスバリデーションの設定
        params: ハイパーパラメータの設定値
        vdrbose: 途中出力メッセージの単位（学習に影響はない）

    Returns:
        oof_pred: validでの予測結果のnp.array
        models: cv分のLightGBMモデル情報
    """

    # デフォルトのハイパーパラメータ
    LGBM_DEFAULT_PARAMS = {
        "objective": "binary",
        "learning_rate": .1,
        "max_depth": 6,
        "n_estimators": 1000,
        "colsample_bytree": .7,
        "importance_type": "gain"
    }

    # もしハイパーパラメータがセットされてないときは、デフォパラメータを使う
    if params is None:
        params = LGBM_DEFAULT_PARAMS

    # もしcvがセットされていないときは、このcvを使う
    if cv is None:
        cv = StratifiedKFold(n_splits=2, shuffle=True)

    # LightGBMに入力用にDataFrameを変換
    X, y = feat_train_df.values, target_df.values

    models = []
    oof_pred = np.zeros_like(y, dtype=np.float)
    shap_values = None
    expected_values = None

    # sklearn.API
    for i, (idx_train, idx_valid) in enumerate(cv.split(X,y)):
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]

        clf = lgbm.LGBMClassifier(**params)
        clf.fit(x_train, y_train,
                # []内で()が必要なのか・・？
                eval_set=[(x_valid, y_valid)],
                early_stopping_rounds=100,
                eval_metric="logloss",
                verbose=verbose
                )

        pred_i = clf.predict_proba(x_valid)[:, 1]
        oof_pred[idx_valid] = pred_i
        models.append(clf)
        print(f"Fold{i+1} AUC: {roc_auc_score(y_valid, pred_i):.4f}")

        if do_shap:
            explainer = shap.TreeExplainer(clf)

            if shap_values is None:
                shap_values = explainer.shap_values(feat_train_df)
                expected_values = explainer.expected_value
            else:
                tmp_shap = explainer.shap_values(feat_train_df)
                shap_values = [x+y for (x,y) in zip(shap_values, tmp_shap)]
                
                tmp_expected = explainer.expected_value
                expected_values = [x+y for (x,y) in zip(expected_values, tmp_expected)]
                
    score = roc_auc_score(y, oof_pred)
    print(f"FINISHED!!! whole score: {score:.4f}")

    if do_shap:
        shap_values = np.array(shap_values) / (cv.n_splits) 
        expected_values = np.array(expected_values) / (cv.n_splits) 
        # shap.summary_plot(shap_values, feat_train_df)
        return oof_pred, models, shap_values, expected_values

    else:
        return oof_pred, models




# LightGBMでのpredict関数（二値分類）
def create_predict(models, predict_df):
    """
    与えられた機械学習モデル（models）で予測する
    モデルごとの予測結果の平均値を出力
    Args:
        models: cv分のLightGBMモデル情報
        predict_df: 予測用のデータセット(学習時と同じカラム構成)
    Returns:
        pred: モデルごとの予測結果の平均値のnp.array
    """
    pred = np.array([model.predict_proba(predict_df.values)[:,1] for model in models])
    pred = np.mean(pred, axis=0) # axis=0で列ごと
    
    return pred



# valid　と　test　の予測結果の分布状況を可視化（確率密度で）
def valid_test_density_plot(oof_pred, pred):
    """
    検証データと予測データで、結果の分布にズレがないか確認
    Args:
        oof_pred: 検証データでの予測結果のnp.array
        pred: 予測データでの予測結果のnp.array
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data= pd.DataFrame({
        "OOF": pd.Series(oof_pred),
        "Test": pd.Series(pred)}),
        ax=ax, stat="density", common_norm=False)
    ax.set_title("valid と Test での予測結果の分布状況")
    ax.grid()
    fig.tight_layout()
    
    return fig, ax



# feature_importanceの可視化
def visualize_importance(models, feat_train_df, plot_columns = 30):
    """
    LightGBMの model 配列の feature_importance を plot する
    CVごとのブレを boxenplot として表現する
    args:
        models: cv分のLightGBMモデル情報
        feat_train_df: 学習時に使った DataFrame
        plot_columns: TOPいくつまでを表示させるか(int)
    """
    feature_importance_df = pd.DataFrame()

    # "feature_importance"で column数分だけレコードができる（これをcolmunでgroupbyする）
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df["feature_importance"] = model.feature_importances_
        _df["column"] = feat_train_df.columns
        _df["fold"] = i+1
        feature_importance_df = pd.concat([feature_importance_df, _df],
                                axis=0,
                                ignore_index=True)
    
    order = feature_importance_df.groupby("column")\
        .sum()[["feature_importance"]]\
        .sort_values("feature_importance",ascending=False).index[:plot_columns]
    
    fig, ax = plt.subplots(figsize=(8, max(6,len(order)* .2)))
    sns.boxenplot(data=feature_importance_df,
                    x="feature_importance",
                    y="column",
                    orient="h",
                    order=order,
                    ax=ax,
                    palette="viridis")
    ax.tick_params(axis="x", rotation=90)
    ax.grid()
    fig.tight_layout()

    return fig, ax



# ROC曲線の可視化
def roc_curve_plot(target_values, oof_pred):
    """
    ROC曲線の可視化、およびAUCを出力
    Args:
        target_values: 正解ラベルのvalues
        oof_pred: 検証データでの予測結果のnp.array
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    fpr, tpr, _ = roc_curve(target_values, oof_pred)
    ax.plot(fpr, tpr, label="ROC曲線")
    ax.legend()
    ax.plot(np.linspace(0,1), np.linspace(0,1), "--", color="gray")
    fig.tight_layout()
    print(f"AUC: {roc_auc_score(target_values, oof_pred)}")

    return fig, ax




# テーブル同士での各カラムごとの重複度合いをベン図で可視化
def plot_venn2(left, right, column, set_labels, ax=None):
    """
    ２テーブルの対象カラムの重複度合いをベン図で可視化
    Args: 
        left/right: テーブルのDataFrame
        column: 対象カラム名
        set_labels: ベン図上でのラベル名（list）
    """
    left_set = set(left[column])
    right_set = set(right[column])
    venn2(subsets=[left_set,right_set], set_labels=set_labels, ax=ax)
    
    return ax


def right_left_insersection(left_df, right_df, columns, set_labels):
    """
    ２テーブルの共通カラムの重複度合いを、すべてベン図で可視化
    Args: 
        left_df/right_df: テーブルのDataFrame
        column: 対象カラム名
        set_labels: ベン図上でのラベル名（list）
    """
    if columns == "__all__":
        columns = set(left_df.columns) & set(right_df.columns)

    columns = list(columns)
    nfigs = len(columns)
    ncols = 4  # Notebookの使える幅によって調整
    nrows = - (- nfigs//ncols) # 切り上げ処理
    fig, axes = plt.subplots(figsize=(3*ncols, 2*nrows), ncols=ncols, nrows=nrows)
    axes = np.ravel(axes) # １次元にして、うまくfor文でplotできるように

    for column, ax in zip(columns, axes):
        plot_venn2(left_df, right_df, column=column, set_labels=set_labels, ax=ax)
        ax.set_title(column)
    fig.tight_layout()

    return fig, ax



# 作成画像を.png形式で所定場所に保存する
def savefig(fig, to, dir_path):
    """
    出力画像を.png形式で保存する
    画像作成時のfigを引数にしている
    Args:
        input:
            fig: 画像作成時のfig
            to: ファイル名
            dir_path: 保存したいフォルダPATH
    """
    to = os.path.join(dir_path, to + ".png")
    print(f"save to {to} .")
    fig.tight_layout()
    fig.savefig(to, dpi=120) # dot per inch ： レゾリューション（解像度）

