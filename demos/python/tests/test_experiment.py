"""Tests for Experiment tracking class."""

import sys
import os
import tempfile
import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synadb import Experiment, Run, RunStatus


class TestExperimentImport:
    """Test that Experiment classes can be imported."""
    
    def test_import(self):
        """Test basic import."""
        from synadb import Experiment, Run, RunStatus
        assert Experiment is not None
        assert Run is not None
        assert RunStatus is not None


class TestExperimentCreation:
    """Test Experiment creation and basic operations."""
    
    def test_create_experiment(self):
        """Test creating an experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            exp = Experiment('test_exp', db_path)
            assert exp.name == 'test_exp'
            assert exp.path == db_path
            exp.close()
    
    def test_context_manager(self):
        """Test experiment as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with Experiment('test_exp', db_path) as exp:
                assert exp.name == 'test_exp'


class TestRunLifecycle:
    """Test Run creation and lifecycle."""
    
    def test_start_run(self):
        """Test starting a run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with Experiment('test_exp', db_path) as exp:
                run = exp.start_run()
                assert run.id is not None
                assert run.experiment_name == 'test_exp'
                assert run.status == RunStatus.RUNNING
                run.end()
    
    def test_run_with_tags(self):
        """Test starting a run with tags."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with Experiment('test_exp', db_path) as exp:
                run = exp.start_run(tags=['baseline', 'v1'])
                assert run.tags == ['baseline', 'v1']
                run.end()
    
    def test_run_context_manager(self):
        """Test run as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with Experiment('test_exp', db_path) as exp:
                with exp.start_run() as run:
                    assert run.status == RunStatus.RUNNING
                
                # After context exit, should be completed
                assert run.status == RunStatus.COMPLETED
    
    def test_run_context_manager_on_exception(self):
        """Test run marks as failed on exception."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with Experiment('test_exp', db_path) as exp:
                try:
                    with exp.start_run() as run:
                        run_id = run.id
                        raise ValueError("Test error")
                except ValueError:
                    pass
                
                # Should be marked as failed
                assert run.status == RunStatus.FAILED


class TestParameterLogging:
    """Test parameter logging."""
    
    def test_log_param(self):
        """Test logging a single parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with Experiment('test_exp', db_path) as exp:
                with exp.start_run() as run:
                    run.log_param('lr', 0.001)
                    assert run.params['lr'] == 0.001
    
    def test_log_params(self):
        """Test logging multiple parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with Experiment('test_exp', db_path) as exp:
                with exp.start_run() as run:
                    run.log_params({'lr': 0.001, 'batch_size': 32})
                    assert run.params['lr'] == 0.001
                    assert run.params['batch_size'] == 32


class TestMetricLogging:
    """Test metric logging."""
    
    def test_log_metric(self):
        """Test logging a single metric."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with Experiment('test_exp', db_path) as exp:
                with exp.start_run() as run:
                    run.log_metric('loss', 0.5)
                    run_id = run.id
                
                # Verify metric was stored
                metrics = exp.get_metrics(run_id, 'loss')
                assert len(metrics) == 1
                assert metrics[0][1] == 0.5
    
    def test_log_metric_with_step(self):
        """Test logging metrics with step numbers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with Experiment('test_exp', db_path) as exp:
                with exp.start_run() as run:
                    for i in range(5):
                        run.log_metric('loss', 1.0 - i * 0.1, step=i)
                    run_id = run.id
                
                # Verify metrics
                metrics = exp.get_metrics(run_id, 'loss')
                assert len(metrics) == 5
                assert metrics[0] == (0, 1.0)
                assert metrics[4] == (4, 0.6)
    
    def test_log_metrics(self):
        """Test logging multiple metrics at once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with Experiment('test_exp', db_path) as exp:
                with exp.start_run() as run:
                    run.log_metrics({'loss': 0.5, 'accuracy': 0.9}, step=0)
                    run_id = run.id
                
                # Verify both metrics
                loss = exp.get_metrics(run_id, 'loss')
                acc = exp.get_metrics(run_id, 'accuracy')
                assert len(loss) == 1
                assert len(acc) == 1
    
    def test_get_metric_tensor(self):
        """Test getting metrics as numpy array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with Experiment('test_exp', db_path) as exp:
                with exp.start_run() as run:
                    for i in range(10):
                        run.log_metric('loss', 1.0 - i * 0.1, step=i)
                    run_id = run.id
                
                # Get as tensor
                tensor = exp.get_metric_tensor(run_id, 'loss')
                assert isinstance(tensor, np.ndarray)
                assert len(tensor) == 10


class TestArtifactLogging:
    """Test artifact logging."""
    
    def test_log_artifact_dict(self):
        """Test logging a dictionary artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with Experiment('test_exp', db_path) as exp:
                with exp.start_run() as run:
                    run.log_artifact('config', {'key': 'value'})
                    run_id = run.id
                
                # Retrieve artifact
                artifact = exp.get_artifact(run_id, 'config')
                assert artifact == {'key': 'value'}
    
    def test_log_artifact_bytes(self):
        """Test logging bytes artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with Experiment('test_exp', db_path) as exp:
                with exp.start_run() as run:
                    run.log_artifact('data', b'raw bytes')
                    run_id = run.id
                
                # Retrieve artifact
                artifact = exp.get_artifact(run_id, 'data')
                assert artifact == b'raw bytes'
    
    def test_list_artifacts(self):
        """Test listing artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with Experiment('test_exp', db_path) as exp:
                with exp.start_run() as run:
                    run.log_artifact('model.pt', {'weights': [1, 2, 3]})
                    run.log_artifact('config.json', {'lr': 0.001})
                    run_id = run.id
                
                # List artifacts
                artifacts = exp.list_artifacts(run_id)
                assert 'model.pt' in artifacts
                assert 'config.json' in artifacts


class TestQueryRuns:
    """Test querying runs."""
    
    def test_query_all_runs(self):
        """Test querying all runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with Experiment('test_exp', db_path) as exp:
                # Create multiple runs
                for i in range(3):
                    with exp.start_run() as run:
                        run.log_param('run_num', i)
                
                # Query all
                runs = exp.query()
                assert len(runs) == 3
    
    def test_query_by_status(self):
        """Test filtering by status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with Experiment('test_exp', db_path) as exp:
                # Create completed run
                with exp.start_run() as run:
                    pass
                
                # Create failed run
                try:
                    with exp.start_run() as run:
                        raise ValueError()
                except ValueError:
                    pass
                
                # Query completed only
                completed = exp.query(filter={'status': 'completed'})
                assert len(completed) == 1
                
                # Query failed only
                failed = exp.query(filter={'status': 'failed'})
                assert len(failed) == 1
    
    def test_query_by_tags(self):
        """Test filtering by tags."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with Experiment('test_exp', db_path) as exp:
                with exp.start_run(tags=['baseline']) as run:
                    pass
                
                with exp.start_run(tags=['experiment']) as run:
                    pass
                
                # Query by tag
                baseline = exp.query(filter={'tags': 'baseline'})
                assert len(baseline) == 1
    
    def test_query_sort_by_metric(self):
        """Test sorting by metric."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with Experiment('test_exp', db_path) as exp:
                # Create runs with different accuracies
                for acc in [0.8, 0.9, 0.85]:
                    with exp.start_run() as run:
                        run.log_metric('accuracy', acc)
                
                # Sort by accuracy descending
                runs = exp.query(sort_by='accuracy', ascending=False)
                
                # Verify order
                accuracies = [exp.get_metric_tensor(r.id, 'accuracy')[-1] for r in runs]
                assert accuracies == sorted(accuracies, reverse=True)


class TestRunRetrieval:
    """Test retrieving specific runs."""
    
    def test_get_run(self):
        """Test getting a specific run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with Experiment('test_exp', db_path) as exp:
                with exp.start_run(tags=['test']) as run:
                    run.log_param('lr', 0.001)
                    run_id = run.id
                
                # Retrieve run
                retrieved = exp.get_run(run_id)
                assert retrieved is not None
                assert retrieved.id == run_id
                assert retrieved.status == RunStatus.COMPLETED
    
    def test_get_nonexistent_run(self):
        """Test getting a run that doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with Experiment('test_exp', db_path) as exp:
                result = exp.get_run('nonexistent-id')
                assert result is None
    
    def test_list_runs(self):
        """Test listing all run IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with Experiment('test_exp', db_path) as exp:
                run_ids = []
                for i in range(3):
                    with exp.start_run() as run:
                        run_ids.append(run.id)
                
                # List runs
                listed = exp.list_runs()
                assert len(listed) == 3
                for rid in run_ids:
                    assert rid in listed


class TestCompareRuns:
    """Test comparing runs."""
    
    def test_compare_runs(self):
        """Test comparing multiple runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with Experiment('test_exp', db_path) as exp:
                run_ids = []
                
                with exp.start_run() as run:
                    run.log_param('lr', 0.001)
                    run.log_metric('accuracy', 0.85)
                    run_ids.append(run.id)
                
                with exp.start_run() as run:
                    run.log_param('lr', 0.01)
                    run.log_metric('accuracy', 0.90)
                    run_ids.append(run.id)
                
                # Compare
                comparison = exp.compare_runs(run_ids)
                
                assert 'runs' in comparison
                assert 'params' in comparison
                assert 'metrics' in comparison
                assert len(comparison['runs']) == 2
                assert 'accuracy' in comparison['metrics']


class TestDeleteRun:
    """Test deleting runs."""
    
    def test_delete_run(self):
        """Test deleting a run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            with Experiment('test_exp', db_path) as exp:
                with exp.start_run() as run:
                    run.log_param('lr', 0.001)
                    run.log_metric('loss', 0.5)
                    run_id = run.id
                
                # Verify run exists
                assert exp.get_run(run_id) is not None
                
                # Delete run
                deleted = exp.delete_run(run_id)
                assert deleted > 0
                
                # Verify run is gone
                assert exp.get_run(run_id) is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
