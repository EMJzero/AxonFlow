class MetaSettings(type):
    # Prevent instantiation of inheriting classes.
    def __call__(cls, *args, **kwargs):
        raise TypeError(f"Instances of {cls.__name__} cannot be created. Use the class directly.")
    
    # Print the forceful update INFO message every time a setting is assigned!
    # If the VERBOSE attribute is present, use it to toggle the print.
    def __setattr__(cls, name, value):
        if getattr(cls, name, None) != value and (not hasattr(cls, "VERBOSE") or cls.VERBOSE) and name != "VERBOSE":
            print(f"INFO: forcefully updating setting {name} to {value}")
        super().__setattr__(name, value)

class Settings(metaclass = MetaSettings):
    # Enable detailed logging of heuristics
    VERBOSE = True