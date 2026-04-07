# KG-RAG4NuclearIE: ナレッジグラフで強化したRAGによる原子力トラブル事例の構造化抽出

東京大学 **出町研究室** にて、**日本原子力研究開発機構（JAEA）** の技術支援のもと取り組んだ、原子力プラントのトラブル事例レポートからの構造化情報抽出パイプラインです。**情報処理学会 第88回全国大会（IPSJ 2026）** にて発表。

> [English README](README_EN.md)

## 課題背景

原子力分野の技術文書からの情報抽出には、非構造化テキスト・多言語混在（EN/JP/FR）・文書間への情報分散といった困難があります。単純なRAGではエビデンスが不完全でハルシネーションのリスクがあるため、ナレッジグラフを活用した **KG-RAG パイプライン**を提案し、**エビデンスの完全性向上**と**制御可能な構造化出力**を実現します。

## パイプライン概要

<p align="center">
  <img src="docs/figures/pipeline_overview.png" width="700" alt="KG-RAG Pipeline Overview"/>
</p>

## 手法詳細

### ナレッジグラフ構築

テキストからLLMでトリプル `(主語, 述語, 目的語)` を抽出し、NetworkX MultiDiGraphとしてマージ・重複排除します。

<p align="center">
  <img src="docs/figures/kg_construction.png" width="700" alt="Knowledge Graph Construction"/>
</p>

### KG強化型ハイブリッド検索

BM25（キーワード）+ ベクトル検索（MiniLM-L6-v2）をRRFで融合し、さらにKGのマルチホップ走査（2ホップ）で文書横断的なエビデンスを回収します。

<p align="center">
  <img src="docs/figures/hybrid_retrieval.png" width="700" alt="KG-Enhanced Hybrid Retrieval"/>
</p>

### 根拠付き抽出＆検証

LLMが8項目スキーマに従い構造化抽出。各フィールドをN-gramマッチングでエビデンスと照合し、**Supported** のみ最終出力に保持します。

<p align="center">
  <img src="docs/figures/extraction_verification.png" width="700" alt="Grounded Extraction & Verification"/>
</p>

## 8項目構造化スキーマ

| # | フィールド | 説明 |
|---|---|---|
| 1 | `facility_name` | 当該施設名 |
| 2 | `event_date` | トラブルの発生日時 |
| 3 | `event_location` | トラブル発生箇所（系統、機器名称 等） |
| 4 | `event_description` | トラブルの内容及び状況（漏えい量/速度、破損規模等） |
| 5 | `cause` | トラブル原因（溶接不良、熱応力による亀裂進展 等） |
| 6 | `detection_method` | トラブル発見方法 |
| 7 | `plant_status` | トラブル発生時の運転状態 |
| 8 | `response` | トラブル後の対応（補修方法、再稼働時期） |

## 実験設定

| 項目 | 値 |
|---|---|
| コーパス | 高速炉トラブル技術報告書 7件（~150ページ） |
| チャンク | 621（512トークン、オーバーラップ50） |
| ナレッジグラフ | 181エンティティ、123リレーション |
| 比較手法 | RAG（ベースライン） vs KG-RAG（提案手法） |
| LLM | GPT-4-turbo（クラウド）/ Llama-3.1-8B（ローカル） |

**評価指標**: Faithfulness（Supported率）、Entity Precision（施設取り違え防止）、Evidence Size（チャンク数）

## 評価結果

| LLM | 手法 | Faithfulness | Entity Precision | #Retrieved |
|---|---|---|---|---|
| GPT-4-turbo | RAG | 36.4% | 50.0% | 5 |
| GPT-4-turbo | **KG-RAG** | **91.7%** | **100.0%** | 10 |
| Llama 3.1-8B | RAG | 100.0% | 50.0% | 5 |
| Llama 3.1-8B | **KG-RAG** | **100.0%** | **100.0%** | 10 |

### 主な知見

- **KG拡張がエビデンスカバレッジを2倍に** — 2ホップ走査で文書横断的エビデンスを集約（5 → 10チャンク）
- **エンティティ対応検索が施設間混同を排除** — コーパス外クエリ（EBR-I）に対しnull抽出を正しく返却（Vanilla RAGは14件の不正確クレーム生成）
- **GPT-4 vs Llama: 対照的なハルシネーションパターン** — GPT-4は創造的だがハルシネーション発生（36.4%）、Llamaは保守的だがエンティティ誤認（Ent.P. 50%）
- **KG-RAGは両モデルでEntity Precisionを50% → 100%に改善**

---

## プロジェクト構成

```text
src/
  document_loader.py      -- Docling文書解析＆テキストチャンク分割
  knowledge_graph.py      -- LLMトリプル抽出＆NetworkXグラフ構築
  retriever.py            -- BM25 + ベクトル + RRF + KG拡張
  extractor.py            -- 8項目構造化抽出＆N-gramクレーム検証
  llm_client.py           -- OpenAI / Llama 統合LLMラッパー
  pipeline.py             -- エンドツーエンドパイプライン統合
  main.py                 -- CLIエントリーポイント

config/
  default.yaml            -- パイプライン全パラメータ
```

## 実行方法

```bash
pip install -e .
cp .env.example .env   # OPENAI_API_KEY を記入
# PDF文書を data/sample_docs/ に配置

kg-rag -q "What coolant leak incident occurred at DFR?"   # 単一クエリ
kg-rag                                                      # 対話モード
kg-rag -q "DFR coolant leak details" -o results.json        # JSON出力
```

```python
from src.pipeline import KGRAGPipeline, PipelineConfig

config = PipelineConfig(input_dir="data/sample_docs", llm_provider="openai", llm_model="gpt-4o")
pipeline = KGRAGPipeline(config)
pipeline.index()
results = pipeline.query("What incident occurred at DFR?")
```

## 技術スタック

| コンポーネント | 技術 |
|---|---|
| ナレッジグラフ | NetworkX (MultiDiGraph) |
| 疎検索 / 密検索 | BM25 (rank-bm25) / Sentence-Transformers (MiniLM-L6-v2) |
| ランク融合 | Reciprocal Rank Fusion (RRF) |
| LLM | OpenAI API (GPT-4-turbo) / Llama-3.1-8B (local) |
| 文書解析 | Docling |
| クレーム検証 | 文字N-gram重複スコアリング |

## 参考文献

- P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," *NeurIPS*, 2020.
- Z. Ji et al., "Survey of Hallucination in Natural Language Generation," *ACM Computing Surveys*, 2023.
- J.L. Phillips, "Full Power Operation of the Dounreay Fast Reactor," *ANS-100*, 1965.
- J.L. Phillips, "Operating Experience with the Dounreay Fast Reactor-2," *Nuclear Power*, 1962.
- R.R. Matthews et al., "Location and Repair of the DFR Leak," *Nuclear Engineering*, 1968.
