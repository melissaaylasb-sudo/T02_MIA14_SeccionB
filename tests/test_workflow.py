"""Pruebas de contratos y aislamiento. No entrenan modelos ni usan datos de tesis."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from src.experimento import inspect_plan, load_model_data, read_config, run_experiment
from src.ingesta import sha256_file
from src.importar_matlab import load_matlab_samples
from src.modelos import build_pipeline, candidate_specs
from src.particiones import make_splits, nested_plan
from src.validacion import serializable_plan

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.config, _ = read_config(ROOT / "config" / "model_config.yml")
        self.protocol = {
            "reviewed": True, "rationale": "Fixture de índices para probar aislamiento",
            "strategy": "group_kfold", "seed": 42, "outer_splits": 3, "inner_splits": 2,
        }
        # Valores arbitrarios para verificar índices; no son una campaña simulada.
        self.X = pd.DataFrame({"fixture_feature": np.arange(24, dtype=float)})
        self.groups = np.repeat(np.arange(6), 4)

    def test_default_inspection_does_not_fit(self):
        with patch.object(GridSearchCV, "fit", side_effect=AssertionError("fit prohibido")):
            result = inspect_plan(self.config)
        self.assertEqual(result["status"], "pending_real_data_and_protocol")
        self.assertFalse(result["training_enabled"])

    def test_training_disabled_before_data_access(self):
        with patch("src.experimento.load_model_data", side_effect=AssertionError("No leer datos")):
            with self.assertRaisesRegex(ValueError, "deshabilitado"):
                run_experiment(self.config, "fixture")

    def test_candidates_are_unfitted_pipelines(self):
        for name, (pipeline, _) in candidate_specs(self.config).items():
            self.assertFalse(hasattr(pipeline.named_steps["imputer"], "statistics_"))
            self.assertEqual(pipeline.named_steps["imputer"].strategy, "median")
            self.assertIn("variance", pipeline.named_steps)
            self.assertIn("scale", pipeline.named_steps)
            self.assertIn("model", pipeline.named_steps)
        self.assertNotIn("neural_network", candidate_specs(self.config))
        self.assertEqual(len(candidate_specs(self.config)), 6)
        self.assertIn("support_vector", candidate_specs(self.config))
        self.assertEqual(build_pipeline("neural_network", 42).named_steps["model"].hidden_layer_sizes, (16,))

    def test_unknown_family_and_parameters_rejected(self):
        with self.assertRaises(ValueError):
            build_pipeline("unknown", 42)
        config = copy.deepcopy(self.config)
        config["models"]["ridge"]["grid"] = {"model__does_not_exist": [1]}
        with self.assertRaises(ValueError):
            candidate_specs(config)

    def test_groups_disjoint_in_outer_and_inner_folds(self):
        plan = nested_plan(self.X, self.protocol, self.groups)
        all_test = []
        for p in plan:
            train, test = p["train"], p["test"]
            self.assertFalse(set(self.groups[train]) & set(self.groups[test]))
            all_test.extend(test)
            for a, b in p["inner"]:
                self.assertFalse(set(self.groups[train[a]]) & set(self.groups[train[b]]))
                self.assertFalse(set(train[a]) & set(test))
                self.assertFalse(set(train[b]) & set(test))
        self.assertEqual(sorted(all_test), list(range(len(self.X))))

    def test_unreviewed_protocol_is_blocked(self):
        protocol = {**self.protocol, "reviewed": False}
        with self.assertRaisesRegex(ValueError, "Revise"):
            nested_plan(self.X, protocol, self.groups)

    def test_insufficient_groups_in_inner_fold_is_blocked(self):
        protocol = {**self.protocol, "inner_splits": 5}
        with self.assertRaisesRegex(ValueError, "grupos"):
            nested_plan(self.X, protocol, self.groups)

    def test_unavailable_training_feature_is_blocked(self):
        X = self.X.copy()
        X["absent"] = np.nan
        with self.assertRaisesRegex(ValueError, "sin observaciones"):
            nested_plan(X, self.protocol, self.groups)

    def test_no_variation_is_blocked(self):
        with self.assertRaisesRegex(ValueError, "constantes"):
            nested_plan(self.X * 0, self.protocol, self.groups)

    def test_kfold_reproducible_and_rejects_groups(self):
        a = make_splits(24, 3, "kfold", 42)
        b = make_splits(24, 3, "kfold", 42)
        for first, second in zip(a, b):
            np.testing.assert_array_equal(first[1], second[1])
        with self.assertRaises(ValueError):
            make_splits(24, 3, "kfold", 42, self.groups)

    def test_saved_inner_indices_are_global(self):
        plan = nested_plan(self.X, self.protocol, self.groups)
        saved = serializable_plan(plan, np.arange(100, 124))
        for p, record in zip(plan, saved):
            self.assertEqual(record["test_source_records"], (100 + p["test"]).tolist())
            for split in record["inner"]:
                self.assertTrue(set(split["train_positions"]) <= set(record["train_positions"]))
                self.assertTrue(set(split["validation_positions"]) <= set(record["train_positions"]))

    def test_model_input_excludes_identifiers_and_detects_tampering(self):
        # Fixture mínima de contrato de archivo; no se ajusta ningún estimador.
        with tempfile.TemporaryDirectory(prefix="pantographic-contract-") as temp:
            root = Path(temp)
            folder = root / "data" / "processed" / "fixture"
            folder.mkdir(parents=True)
            table = folder / "model_table.csv"
            table.write_text(
                "__source_record,id,group,feature,response\n1,001,A,2,3\n2,002,B,4,5\n",
                encoding="utf-8",
            )
            roles = {"target": "response", "identifiers": ["id"],
                     "predictors": ["feature"], "groups": ["group"]}
            schema = {
                "project": {"task": "regression", "target_unit": "fixture_unit"},
                "columns": roles, "predictor_units": {"feature": "fixture_unit"},
                "curation": {"drop_exact_duplicates": False, "missing_values": {}},
                "rules": {"target_imputation": False, "identifier_as_predictor": False,
                          "fit_statistical_preprocessing_before_split": False},
            }
            (folder / "data_quality_report.json").write_text(json.dumps({
                "schema": schema, "roles": roles, "output_sha256": sha256_file(table),
            }), encoding="utf-8")
            config = copy.deepcopy(self.config)
            config["data"]["model_table"] = str(table)
            config["protocol"].update(strategy="group_kfold", group_columns=["group"])
            with patch("src.experimento.ROOT", root):
                X, y, trace, groups, *_ = load_model_data(config)
                self.assertEqual(X.columns.tolist(), ["feature"])
                self.assertEqual(trace["id"].tolist(), ["001", "002"])
                self.assertEqual(len(set(groups)), 2)
                self.assertEqual(y.name, "response")
                table.write_text(table.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "hash"):
                    load_model_data(config)

    def test_matlab_samples_parser_reads_only_declared_numeric_table(self):
        source = """
        Case_n = (1:3)';
        geometry = [1.5; 2.5; 3.5];
        response = [10; NaN; 30];
        constant = repmat(0.5,3,1);
        ignored = system('echo this must never run');
        samples = table(Case_n, geometry, response, constant);
        """
        with tempfile.TemporaryDirectory(prefix="pantographic-matlab-") as temp:
            path = Path(temp) / "fixture.m"
            path.write_text(source, encoding="utf-8")
            frame = load_matlab_samples(path)
        self.assertEqual(frame.columns.tolist(), ["Case_n", "geometry", "response", "constant"])
        self.assertEqual(frame["Case_n"].tolist(), [1, 2, 3])
        self.assertTrue(np.isnan(frame.loc[1, "response"]))
        self.assertEqual(frame["constant"].tolist(), [0.5, 0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
