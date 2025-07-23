class MetaSettings(type):
    # Prevent instantiation of inheriting classes.
    def __call__(cls, *args, **kwargs):
        raise TypeError(f"Instances of {cls.__name__} cannot be created. Use the class directly.")
    
    # Print the forceful update INFO message every time a setting is assigned!
    # If the VERBOSE attribute is present, use it to toggle the print.
    def __setattr__(cls, name, value):
        if getattr(cls, name, None) != value and (not hasattr(cls, "VERBOSE") or cls.VERBOSE) and name != "VERBOSE" and name != "VERBOSE_COLOR":
            print(f"INFO: forcefully updating setting {name} to {value}")
        super().__setattr__(name, value)

class Settings(metaclass = MetaSettings):
    # Enable detailed logging of "@core" heuristics
    VERBOSE = True
    # Enable timing of "@core" heuristics (requires 'VERBOSE = True')
    TIMING = True
    # Color for the information printed by 'VERBOSE' and 'TIMING'
    # => can be a string among 'termcolor.COLORS' or an (R, G, B) tuple
    VERBOSE_COLOR = "yellow"
    # Default print interval in seconds for timer-controller prints
    PRINT_INTERVAL = 60
    # If True, 'spawn_process' spawns a new process, otherwise it runs sequentially
    MULTIPROCESSING = True
    # Maximum execution time allowed for processes, after which they are killed with an exception
    # => set to a number of seconds or 'None' to execute until completion
    MULTIPROCESSING_TIMEOUT = None
    # Timeout in seconds for the join or the spawning of processes.
    MULTIPROCESSING_SPINNING_INTERVAL = 0.5
    # Sets the maximum number of processes that will be used concurrently
    PROCESSES_COUNT = 4