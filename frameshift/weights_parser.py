from typing import Dict

def parse_object_weights(weights_str: str) -> Dict[str, float]:
    """
    Parses a weights string (e.g., "face:1.0,person:0.8,default:0.5")
    into a dictionary mapping labels to float weights.
    """
    weights: Dict[str, float] = {}
    # Establish a baseline default weight, which can be overridden by 'default:value' in the string
    internal_default_weight = 0.5

    if not weights_str:
        # Fallback to internal defaults if string is empty or None
        weights = {'face': 1.0, 'person': 0.8, 'default': internal_default_weight}
        return weights

    pairs = weights_str.split(',')
    found_default_in_string = False
    for pair in pairs:
        parts = pair.split(':', 1) # Split only on the first colon
        if len(parts) == 2:
            label = parts[0].strip().lower() # Normalize label to lowercase
            if not label: # Skip if label is empty
                print(f"Warning: Empty label in --object_weights, skipping pair: {pair}")
                continue
            try:
                weight = float(parts[1].strip())
                if weight < 0:
                    print(f"Warning: Negative weight for '{label}' not allowed, using 0.0: {parts[1]}")
                    weight = 0.0
                weights[label] = weight
                if label == 'default':
                    found_default_in_string = True
                    internal_default_weight = weight # Update our running default
            except ValueError:
                print(f"Warning: Could not parse weight for '{label}' in --object_weights, skipping: {parts[1]}")
        elif pair.strip(): # If pair is not just whitespace and not a valid pair
            print(f"Warning: Malformed pair in --object_weights, skipping: {pair}")

    # Ensure 'default' weight is present in the final map
    if 'default' not in weights:
        weights['default'] = internal_default_weight
    elif not found_default_in_string: # If 'default' was added but not from string, ensure it's our internal_default_weight
        weights['default'] = internal_default_weight


    # print(f"DEBUG: Parsed weights: {weights}")
    return weights
