from roboflow import Roboflow

# Configuration
API_KEY = "P5W2awqHgQZgmiKPJIu7"
PROJECTS_AND_VERSIONS = [
    {"project_name": "people-detection-o4rdr", "version_number": 7},
    {"project_name": "gempa", "version_number": 11},  # Add more as needed
]
MINIMUM_CONFIDENCE = 0.45

# Initialize Roboflow models
rf = Roboflow(api_key=API_KEY)
models = [
    rf.workspace().project(pv["project_name"]).version(pv["version_number"]).model
    for pv in PROJECTS_AND_VERSIONS
]

def pred_single(img_path):
    """
    Perform a single prediction on the given image using multiple models.

    Args:
        img_path (str): The path to the image to perform detection on.

    Returns:
        tuple: A tuple containing (xmin, ymin, xmax, ymax, class) if a detection 
               with confidence above the minimum threshold is found. Otherwise, returns None.
    """
    for model in models:
        try:
            response = model.predict(img_path).json()
            predictions = response['predictions']

            if predictions:
                for pred in predictions:
                    conf = pred['confidence']
                    if conf > MINIMUM_CONFIDENCE:
                        xmin = pred['x'] - pred['width'] / 2
                        ymin = pred['y'] - pred['height'] / 2
                        xmax = pred['x'] + pred['width'] / 2
                        ymax = pred['y'] + pred['height'] / 2
                        cls = pred['class']
                        return round(xmin), round(ymin), round(xmax), round(ymax), cls
        except Exception as e:
            print(f"Error during prediction with model from project {model.project_name}, version {model.version}: {e}")
    return None

def pred_multiple(img_path):
    """
    Perform prediction on the given image using multiple models and return all detections.

    Args:
        img_path (str): The path to the image to perform detection on.

    Returns:
        list: A list of tuples, each containing (xmin, ymin, xmax, ymax, class) for detections
              with confidence above the minimum threshold.
    """
    detections = []
    for model in models:
        try:
            response = model.predict(img_path).json()
            predictions = response['predictions']

            for pred in predictions:
                conf = pred['confidence']
                if conf > MINIMUM_CONFIDENCE:
                    xmin = pred['x'] - pred['width'] / 2
                    ymin = pred['y'] - pred['height'] / 2
                    xmax = pred['x'] + pred['width'] / 2
                    ymax = pred['y'] + pred['height'] / 2
                    cls = pred['class']
                    detections.append((round(xmin), round(ymin), round(xmax), round(ymax), cls))
        except Exception as e:
            print(f"Error during prediction with model from project {model.project_name}, version {model.version}: {e}")

    return detections
