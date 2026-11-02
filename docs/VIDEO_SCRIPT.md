# Video script — 4 members × 1 minute

Camera on, screen shared on the Streamlit app. ~130 spoken words each. Open the right tab before you start.

---

## Member 1 — Data Versioning (tab ①)

*[Overview tab, architecture diagram]*
"We built an end-to-end pipeline that predicts whether an Indian domestic flight will be delayed by more than
fifteen minutes. I'll cover the first stage: data versioning.

*[Tab ① — version history]* Every dataset is hashed with MD5 and stored once, and Git only keeps a small pointer
file — exactly how DVC works. You can see four versions: the raw extract, a cleaned version, a Q3 refresh that
added a new carrier and a new column, and the full year.

*[Compare v1.0 vs v3.0]* The diff shows duplicates removed, nulls fixed and the schema change.

*[Reproducibility test, 400 rows edited]* Here's why it matters: same version, same hash, identical AUC.
If someone silently edits the file, the metric changes and nobody knows why — but the hash flags it instantly.
Now, how do we turn this data into features?"

---

## Member 2 — Feature Store (tab ②)

"I built the feature store. *[Feature view cards]* Features are defined once, in three views: airport,
carrier and route, each with a time-to-live.

*[Point-in-time section, pick a flight]* The key idea is a point-in-time join. For this flight, the model may only
see values available at midnight. A naive join on the date also includes flights later that evening — and this
flight's own outcome. That's leakage.

*[Click leakage experiment]* The result: the naive model looks like 0.78 AUC offline, but in production it
collapses to about 0.63 — worse than our honest model at 0.67.

*[Online store lookup]* Finally, we materialize the latest values into an online store, so serving uses exactly
the same definitions as training. No training–serving skew.
Next, how we track experiments."

---

## Member 3 — Experiment Tracking & Registry (tab ③)

"I handled experiment tracking with MLflow. *[Launch form]* Let me train a gradient boosting model on data
version 2.0. *[Click Train]* Every run logs its parameters, metrics, the model with its signature, and — crucially —
the dataset version and its MD5 hash, so any result can be traced back to its exact data.

*[Run history and charts]* Here are all our runs. Note we only compare models evaluated on the same data
version — otherwise the holdout windows differ. The leaky run is tagged unsafe and rejected.

*[Registry]* The best run is registered as a model version and promoted with the champion alias. Deployment
never copies files; it just moves the alias, which also makes rollback a one-click operation.
Now let's serve the champion."

---

## Member 4 — Prediction Service & Lineage (tab ④)

"I built the prediction service. *[Champion KPIs]* It resolves the model through the registry alias — here,
version one, trained on data v2.0.

*[Change origin to DEL, hour 19, raise weather]* When I score a flight, the service fetches fresh features from
the online store and returns the delay probability. The what-if chart shows risk building through the evening.

*[Lineage graph]* Every prediction carries its full lineage: dataset hash, feature timestamps, MLflow run and
model version — so any decision can be audited.

*[Batch scoring]* For operations, we can score tomorrow's whole schedule and flag high-risk flights.
The same logic is exposed as a FastAPI REST endpoint.

To conclude: versioning, feature stores, tracking and governed serving turn a model into a reliable,
reproducible system. Thank you."
