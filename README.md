# FINDETECT AML Compliance System

FINDETECT is an end-to-end Anti-Money Laundering (AML) risk analysis pipeline that combines supervised machine learning, anomaly detection, behavioral risk analysis, graph-based exposure analysis, risk scoring, and SAR draft generation.

The integrated pipeline was executed successfully on the full FINDETECT final dataset containing **155,493,482 transactions across 85 days**.

## Architecture

```text
Final Transaction Dataset
        |
        v
+-------------------------------+
|       Feature Dataset         |
+-------------------------------+
        |
        +-------------------+-------------------+
        |                   |                   |
        v                   v                   v
    XGBoost            Autoencoder         Behavioral
  Supervised ML       Anomaly Model        Risk Model
        |                   |                   |
        +-------------------+-------------------+
                            |
                            v
                    Weighted Risk Fusion
                 XGB 40% + AE 30% + BEH 30%
                            |
                            v
                  Silent-Mule Graph Analysis
                            |
                            v
                       Risk Agent
                            |
                            v
                    SAR Draft Generation
