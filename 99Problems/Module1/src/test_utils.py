import unittest
import pandas as pd
import os
import shutil
import time
from unittest.mock import patch, MagicMock

from Module1.src.utils import *



class TestDataUtilities(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.test_dir = 'temp_test_dir'
        cls.test_file_path = os.path.join(cls.test_dir, 'test_data.csv')
        cls.test_df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
        os.makedirs(cls.test_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_save_and_load_csv(self):
        save_to_csv(self.test_df, self.test_file_path)
        loaded_df = load_csv_to_df(self.test_file_path)
        
        pd.testing.assert_frame_equal(self.test_df, loaded_df)


    def test_describe_numeric_col(self):
        data = pd.Series([10.0, 20.0, 30.0, float('nan'), 50.0])
        stats = describe_numeric_col(data)
        
        self.assertEqual(stats["Count"], 4)
        self.assertEqual(stats["Missing"], 1)
        self.assertAlmostEqual(stats["Mean"], 27.5) 
        self.assertEqual(stats["Min"], 10.0)
        self.assertEqual(stats["Max"], 50.0)

    def test_impute_missing_values_median(self):
        data = pd.Series([3, 50, 100, None])
        imputed_data = impute_missing_values(data, method="median")
        
        pd.testing.assert_series_equal(
            imputed_data, pd.Series([3.0, 50.0, 100.0, 50.0]), check_names=False
        )
        
    def test_impute_missing_values_mean(self):
        data = pd.Series([3, 50, 100, None])
        imputed_data = impute_missing_values(data, method="mean")
        
        pd.testing.assert_series_equal(
            imputed_data, pd.Series([3.0, 50.0, 100.0, 51.0]), check_names=False
        )

    def test_create_dummy_cols_logic(self):
        df = pd.DataFrame({'feature': [1, 2, 3], 'category': ['A', 'B', 'A']})

        df['category'] = pd.Categorical(df['category'], categories=['A', 'B'])
        
        processed_df = create_dummy_cols(df, 'category')

        expected_df = pd.DataFrame({
            'feature': [1, 2, 3],
            'category_B': [False, True, False]
        })
        
        pd.testing.assert_frame_equal(
            processed_df.reset_index(drop=True),
            expected_df.reset_index(drop=True)
        )

    
    @patch('time.sleep', return_value=None)
    @patch('mlflow.tracking.client.MlflowClient')
    @patch('mlflow.entities.model_registry.model_version_status.ModelVersionStatus')
    def test_wait_until_ready(self, MockModelVersionStatus, MockMlflowClient, mock_sleep):
        
        MockModelVersionStatus.READY = 'READY_STATUS_OBJECT'
            
        mock_client_instance = MockMlflowClient.return_value
        mock_client_instance.get_model_version.side_effect = [
            MagicMock(status='PENDING'), 
            MagicMock(status='PENDING'), 
            MagicMock(status='READY'),
        ]

   
        MockModelVersionStatus.from_string.side_effect = [
            'PENDING_STATUS_OBJECT',
            'PENDING_STATUS_OBJECT',
            MockModelVersionStatus.READY, 
        ]
    
        def wait_until_ready(model_name, model_version):
            client = MockMlflowClient()
            for _ in range(10):
                model_version_details = client.get_model_version(
                    name=model_name,
                    version=model_version,
                )
                status = MockModelVersionStatus.from_string(model_version_details.status)
                if status == MockModelVersionStatus.READY:
                    break
                time.sleep(1) 
        
        wait_until_ready("test_model", "1")
        
        self.assertEqual(mock_client_instance.get_model_version.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)


    @patch('time.sleep', return_value=None)
    @patch('mlflow.tracking.MlflowClient')
    def test_wait_for_deployment(self, MockMlflowClient, mock_sleep):
        
        mock_details_wrong = {'current_stage': 'Archived'}
        mock_details_correct = {'current_stage': 'Staging'}

        mock_client_instance = MockMlflowClient.return_value
        
        mock_client_instance.get_model_version.side_effect = [
            mock_details_wrong,
            mock_details_wrong,
            mock_details_correct,
        ]

        def wait_for_deployment(model_name, model_version, stage='Staging'):
            client = MockMlflowClient() 
            status = False
            while not status:
                model_version_details = dict(
                    client.get_model_version(name=model_name,version=model_version)
                    )
                if model_version_details['current_stage'] == stage:
                    status = True
                    break
                else:
                    time.sleep(2)
            return status

        result = wait_for_deployment("test_model", "1", stage='Staging')
        
        self.assertTrue(result)
        self.assertEqual(mock_client_instance.get_model_version.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)


# To run use: python -m unittest discover Module1