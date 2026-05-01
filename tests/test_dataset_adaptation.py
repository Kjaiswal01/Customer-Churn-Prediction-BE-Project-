from __future__ import annotations

import unittest

try:
    import pandas as pd
    from enterprise_service import infer_missing_business_signals, prepare_company_dataset
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    pd = None
    infer_missing_business_signals = prepare_company_dataset = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class DatasetAdaptationTests(unittest.TestCase):
    def setUp(self) -> None:
        if IMPORT_ERROR is not None:
            self.skipTest(f"Dataset adaptation dependencies unavailable in the current runtime: {IMPORT_ERROR}")

    def test_telco_dataset_derives_missing_business_signals(self) -> None:
        raw_df = pd.DataFrame(
            [
                {
                    "customerID": "7590-VHVEG",
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "tenure": 1,
                    "PhoneService": "No",
                    "InternetService": "DSL",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 29.85,
                    "TotalCharges": "29.85",
                    "Churn": "No",
                }
            ]
        )

        prepared_df, schema = prepare_company_dataset(raw_df)
        enriched_df, enriched_schema = infer_missing_business_signals(prepared_df, schema)

        for column in [
            "Subscription_Type",
            "Usage_Score",
            "Support_Tickets",
            "Payment_Delay_Days",
            "Feedback",
            "Issue_Category",
            "Last_Interaction_Days",
        ]:
            self.assertIn(column, enriched_df.columns)

        self.assertGreaterEqual(enriched_schema.get("signal_column_count", 0), 6)
        self.assertGreaterEqual(enriched_schema.get("usable_feature_count", 0), 8)


if __name__ == "__main__":
    unittest.main()
