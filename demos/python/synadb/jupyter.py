"""
Jupyter magic commands for SynaDB.

This module provides IPython magic commands for interacting with SynaDB
databases directly from Jupyter notebooks.

Usage:
    %load_ext synadb

    %syna_info mydb.db
    %syna_search vectors.db "machine learning" --k 5
    %syna_plot experiments.db run_001 --metric loss
    %syna_keys mydb.db --pattern "sensor/*"
    %syna_compare experiments.db run_001 run_002 --metric accuracy

Example:
    >>> # In a Jupyter notebook cell:
    >>> %load_ext synadb
    >>> %syna_info mydb.db
    Database: mydb.db
    Keys: 42
    Sample keys: ['sensor/temp', 'sensor/humidity', ...]

_Requirements: 14.5_
"""

try:
    from IPython.core.magic import Magics, magics_class, line_magic
    from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
    from IPython.display import display, HTML
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
    # Create dummy classes for when IPython is not available
    class Magics:
        def __init__(self, shell=None):
            pass
    
    def magics_class(cls):
        return cls
    
    def line_magic(func):
        return func
    
    def magic_arguments():
        def decorator(func):
            return func
        return decorator
    
    def argument(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def parse_argstring(func, line):
        return None


@magics_class
class SynaMagics(Magics):
    """
    Jupyter magic commands for SynaDB.

    Usage:
        %load_ext synadb

        %syna_info mydb.db
        %syna_search vectors.db "machine learning" --k 5
        %syna_plot experiments.db run_001 --metric loss
        %syna_keys mydb.db --pattern "sensor/*"
        %syna_compare experiments.db run_001 run_002 --metric accuracy
    
    _Requirements: 14.5_
    """

    def __init__(self, shell):
        """Initialize the magic commands.
        
        Args:
            shell: IPython shell instance.
        """
        super().__init__(shell)
        self._dbs = {}

    @line_magic
    @magic_arguments()
    @argument('path', help='Database path')
    def syna_info(self, line):
        """
        Show database info.
        
        Displays the number of keys and sample keys from the database.
        
        Usage:
            %syna_info mydb.db
        
        Args:
            path: Path to the database file.
        
        Example:
            >>> %syna_info mydb.db
            Database: mydb.db
            Keys: 42
            Sample keys: ['sensor/temp', 'sensor/humidity', ...]
        """
        args = parse_argstring(self.syna_info, line)
        
        try:
            from .wrapper import SynaDB
            
            with SynaDB(args.path) as db:
                keys = db.keys()
                
                print(f"Database: {args.path}")
                print(f"Keys: {len(keys)}")
                
                if keys:
                    sample = keys[:10] if len(keys) > 10 else keys
                    print(f"Sample keys: {sample}")
                    
                    if len(keys) > 10:
                        print(f"  ... and {len(keys) - 10} more")
        except Exception as e:
            print(f"Error: {e}")

    @line_magic
    @magic_arguments()
    @argument('path', help='Database path')
    @argument('query', help='Search query text')
    @argument('--k', type=int, default=5, help='Number of results')
    @argument('--dimensions', type=int, default=768, help='Vector dimensions')
    def syna_search(self, line):
        """
        Search vectors by text query.
        
        Note: This requires an embedding model to convert text to vectors.
        Currently displays a placeholder message. For actual vector search,
        use the VectorStore API directly with pre-computed embeddings.
        
        Usage:
            %syna_search vectors.db "machine learning" --k 5
        
        Args:
            path: Path to the vector database file.
            query: Search query text.
            --k: Number of results to return (default: 5).
            --dimensions: Vector dimensions (default: 768).
        
        Example:
            >>> %syna_search vectors.db "machine learning" --k 5
            Searching vectors.db for 'machine learning' (k=5)
            Note: Text-to-vector embedding not configured.
        """
        args = parse_argstring(self.syna_search, line)
        
        print(f"Searching {args.path} for '{args.query}' (k={args.k})")
        print("Note: Text-to-vector embedding not configured.")
        print("For vector search, use VectorStore.search() with pre-computed embeddings:")
        print()
        print("  from synadb import VectorStore")
        print("  store = VectorStore(path, dimensions=768)")
        print("  results = store.search(query_embedding, k=5)")

    @line_magic
    @magic_arguments()
    @argument('path', help='Database path')
    @argument('run_id', help='Run ID')
    @argument('--metric', default='loss', help='Metric to plot')
    @argument('--figsize', default='10,6', help='Figure size as width,height')
    @argument('--title', default=None, help='Custom plot title')
    def syna_plot(self, line):
        """
        Plot experiment metrics.
        
        Creates a line plot of the specified metric over training steps.
        Requires matplotlib to be installed.
        
        Usage:
            %syna_plot experiments.db run_001 --metric loss
            %syna_plot experiments.db run_001 --metric accuracy --figsize 12,8
        
        Args:
            path: Path to the experiments database file.
            run_id: Run ID to plot metrics for.
            --metric: Metric name to plot (default: 'loss').
            --figsize: Figure size as 'width,height' (default: '10,6').
            --title: Custom plot title (default: 'Run {run_id}: {metric}').
        
        Example:
            >>> %syna_plot experiments.db abc123 --metric loss
            # Displays a matplotlib plot of the loss metric
        """
        args = parse_argstring(self.syna_plot, line)

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Error: matplotlib not installed")
            print("Install with: pip install matplotlib")
            return

        try:
            from .experiment import Experiment
            
            # Parse figure size
            try:
                width, height = map(float, args.figsize.split(','))
                figsize = (width, height)
            except ValueError:
                figsize = (10, 6)
            
            # Get experiment name from run metadata
            # We need to find which experiment this run belongs to
            from .wrapper import SynaDB
            
            with SynaDB(args.path) as db:
                keys = db.keys()
                
                # Find the experiment name for this run
                experiment_name = None
                for key in keys:
                    if f"/run/{args.run_id}/meta" in key:
                        # Extract experiment name from key: exp/{name}/run/{id}/meta
                        parts = key.split('/')
                        if len(parts) >= 2:
                            experiment_name = parts[1]
                            break
                
                if not experiment_name:
                    print(f"Error: Run '{args.run_id}' not found")
                    return
            
            # Now use Experiment to get metrics
            exp = Experiment(experiment_name, args.path)
            metrics = exp.get_metrics(args.run_id, args.metric)
            
            if not metrics:
                print(f"No metrics found for '{args.metric}' in run '{args.run_id}'")
                return
            
            steps, values = zip(*metrics)
            
            plt.figure(figsize=figsize)
            plt.plot(steps, values, marker='o', markersize=3, linewidth=1.5)
            plt.xlabel('Step')
            plt.ylabel(args.metric)
            
            title = args.title or f'Run {args.run_id}: {args.metric}'
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            exp.close()
            
        except Exception as e:
            print(f"Error: {e}")

    @line_magic
    @magic_arguments()
    @argument('path', help='Database path')
    @argument('--pattern', default=None, help='Key pattern filter (glob-style)')
    @argument('--limit', type=int, default=20, help='Maximum keys to display')
    def syna_keys(self, line):
        """
        List keys in the database.
        
        Displays keys from the database, optionally filtered by a glob pattern.
        
        Usage:
            %syna_keys mydb.db
            %syna_keys mydb.db --pattern "sensor/*"
            %syna_keys mydb.db --limit 50
        
        Args:
            path: Path to the database file.
            --pattern: Optional glob pattern to filter keys.
            --limit: Maximum number of keys to display (default: 20).
        
        Example:
            >>> %syna_keys mydb.db --pattern "sensor/*"
            Keys matching 'sensor/*':
              sensor/temp
              sensor/humidity
              sensor/pressure
        """
        args = parse_argstring(self.syna_keys, line)
        
        try:
            from .wrapper import SynaDB
            import fnmatch
            
            with SynaDB(args.path) as db:
                keys = db.keys()
                
                # Filter by pattern if provided
                if args.pattern:
                    keys = [k for k in keys if fnmatch.fnmatch(k, args.pattern)]
                    print(f"Keys matching '{args.pattern}':")
                else:
                    print(f"All keys ({len(keys)} total):")
                
                # Display keys with limit
                displayed = keys[:args.limit]
                for key in displayed:
                    print(f"  {key}")
                
                if len(keys) > args.limit:
                    print(f"  ... and {len(keys) - args.limit} more")
                    
        except Exception as e:
            print(f"Error: {e}")

    @line_magic
    @magic_arguments()
    @argument('path', help='Database path')
    @argument('run_ids', nargs='+', help='Run IDs to compare')
    @argument('--metric', default='loss', help='Metric to compare')
    def syna_compare(self, line):
        """
        Compare multiple experiment runs.
        
        Displays a comparison table of metrics across multiple runs.
        
        Usage:
            %syna_compare experiments.db run_001 run_002 --metric accuracy
        
        Args:
            path: Path to the experiments database file.
            run_ids: Two or more run IDs to compare.
            --metric: Metric to compare (default: 'loss').
        
        Example:
            >>> %syna_compare experiments.db run_001 run_002 --metric accuracy
            Comparing runs: run_001, run_002
            Metric: accuracy
            
            Run       | Latest | Min    | Max    | Mean
            ----------|--------|--------|--------|--------
            run_001   | 0.95   | 0.65   | 0.95   | 0.82
            run_002   | 0.92   | 0.60   | 0.92   | 0.78
        """
        args = parse_argstring(self.syna_compare, line)
        
        try:
            from .wrapper import SynaDB
            from .experiment import Experiment
            
            # Find experiment name from first run
            with SynaDB(args.path) as db:
                keys = db.keys()
                
                experiment_name = None
                for run_id in args.run_ids:
                    for key in keys:
                        if f"/run/{run_id}/meta" in key:
                            parts = key.split('/')
                            if len(parts) >= 2:
                                experiment_name = parts[1]
                                break
                    if experiment_name:
                        break
                
                if not experiment_name:
                    print(f"Error: No runs found")
                    return
            
            exp = Experiment(experiment_name, args.path)
            
            print(f"Comparing runs: {', '.join(args.run_ids)}")
            print(f"Metric: {args.metric}")
            print()
            
            # Header
            print(f"{'Run':<15} | {'Latest':>8} | {'Min':>8} | {'Max':>8} | {'Mean':>8}")
            print("-" * 15 + "-|-" + "-" * 8 + "-|-" + "-" * 8 + "-|-" + "-" * 8 + "-|-" + "-" * 8)
            
            # Data rows
            for run_id in args.run_ids:
                tensor = exp.get_metric_tensor(run_id, args.metric)
                
                if len(tensor) > 0:
                    latest = tensor[-1]
                    min_val = tensor.min()
                    max_val = tensor.max()
                    mean_val = tensor.mean()
                    
                    print(f"{run_id:<15} | {latest:>8.4f} | {min_val:>8.4f} | {max_val:>8.4f} | {mean_val:>8.4f}")
                else:
                    print(f"{run_id:<15} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8}")
            
            exp.close()
            
        except Exception as e:
            print(f"Error: {e}")

    @line_magic
    @magic_arguments()
    @argument('path', help='Database path')
    @argument('key', help='Key to get value for')
    def syna_get(self, line):
        """
        Get a value from the database.
        
        Retrieves and displays the value for a specific key.
        
        Usage:
            %syna_get mydb.db sensor/temp
        
        Args:
            path: Path to the database file.
            key: Key to retrieve.
        
        Example:
            >>> %syna_get mydb.db sensor/temp
            Key: sensor/temp
            Value: 23.5 (float)
        """
        args = parse_argstring(self.syna_get, line)
        
        try:
            from .wrapper import SynaDB
            
            with SynaDB(args.path) as db:
                # Try different types
                value = db.get_float(args.key)
                if value is not None:
                    print(f"Key: {args.key}")
                    print(f"Value: {value} (float)")
                    return
                
                value = db.get_int(args.key)
                if value is not None:
                    print(f"Key: {args.key}")
                    print(f"Value: {value} (int)")
                    return
                
                value = db.get_text(args.key)
                if value is not None:
                    print(f"Key: {args.key}")
                    print(f"Value: {value} (text)")
                    return
                
                value = db.get_bytes(args.key)
                if value is not None:
                    print(f"Key: {args.key}")
                    print(f"Value: <{len(value)} bytes>")
                    return
                
                print(f"Key '{args.key}' not found")
                
        except Exception as e:
            print(f"Error: {e}")

    @line_magic
    @magic_arguments()
    @argument('path', help='Database path')
    @argument('key', help='Key to get history for')
    @argument('--plot', action='store_true', help='Plot the history')
    def syna_history(self, line):
        """
        Get the history of values for a key.
        
        Retrieves all historical values for a key as a numpy array.
        Optionally plots the history.
        
        Usage:
            %syna_history mydb.db sensor/temp
            %syna_history mydb.db sensor/temp --plot
        
        Args:
            path: Path to the database file.
            key: Key to retrieve history for.
            --plot: If set, plot the history using matplotlib.
        
        Example:
            >>> %syna_history mydb.db sensor/temp
            Key: sensor/temp
            History: [23.5, 24.0, 24.5, 25.0] (4 values)
        """
        args = parse_argstring(self.syna_history, line)
        
        try:
            from .wrapper import SynaDB
            
            with SynaDB(args.path) as db:
                history = db.get_history_tensor(args.key)
                
                if len(history) == 0:
                    print(f"No history found for key '{args.key}'")
                    return
                
                print(f"Key: {args.key}")
                print(f"History: {len(history)} values")
                print(f"  Min: {history.min():.4f}")
                print(f"  Max: {history.max():.4f}")
                print(f"  Mean: {history.mean():.4f}")
                print(f"  Std: {history.std():.4f}")
                
                if args.plot:
                    try:
                        import matplotlib.pyplot as plt
                        
                        plt.figure(figsize=(10, 6))
                        plt.plot(history, marker='o', markersize=3, linewidth=1.5)
                        plt.xlabel('Index')
                        plt.ylabel('Value')
                        plt.title(f'History: {args.key}')
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.show()
                    except ImportError:
                        print("matplotlib not installed, cannot plot")
                        
        except Exception as e:
            print(f"Error: {e}")


def load_ipython_extension(ipython):
    """
    Load the SynaDB extension in IPython.
    
    This function is called when the user runs:
        %load_ext synadb
    
    Args:
        ipython: IPython shell instance.
    """
    if not IPYTHON_AVAILABLE:
        print("Warning: IPython not available. Magic commands will not work.")
        return
    
    ipython.register_magics(SynaMagics)
    print("SynaDB magic commands loaded. Available commands:")
    print("  %syna_info      - Show database info")
    print("  %syna_keys      - List keys in database")
    print("  %syna_get       - Get a value")
    print("  %syna_history   - Get value history")
    print("  %syna_search    - Search vectors")
    print("  %syna_plot      - Plot experiment metrics")
    print("  %syna_compare   - Compare experiment runs")


def unload_ipython_extension(ipython):
    """
    Unload the SynaDB extension from IPython.
    
    This function is called when the user runs:
        %unload_ext synadb
    
    Args:
        ipython: IPython shell instance.
    """
    pass  # No cleanup needed
