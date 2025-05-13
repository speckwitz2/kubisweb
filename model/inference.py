from model.config import cfg as config
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
import numpy as np
import supervision as sv
import time
import cv2

from collections import Counter

def partDetectionCallback(image_slice: np.ndarray) -> sv.Detections:
  try:
    predictor = DefaultPredictor(config)
    output = predictor(image_slice)
    instances = output["instances"].to("cpu")

    if len(instances) == 0:
      return sv.Detections.empty()

    boxes = instances.pred_boxes.tensor.numpy()
    class_id = instances.pred_classes.numpy()
    scores = instances.scores.numpy()

    return sv.Detections(xyxy=boxes, class_id=class_id, confidence=scores)
  except Exception as e:
    print(f"[ERROR in partDetectionCallback] {e}")
    return sv.Detections.empty()


def doDetection(image: np.ndarray, slice_wh=(800, 800)):
  try:
    # time when the function start
    tstart = time.time()
    slicer = sv.InferenceSlicer(
        callback=partDetectionCallback,
        slice_wh=slice_wh,
        overlap_wh=(100, 100),
        overlap_ratio_wh=None,
        iou_threshold=0.5,
        thread_workers=2
    )

    detection = slicer(image)
    metadata = MetadataCatalog.get("cabbage_test_dataset").set(thing_classes=["trash", "cabbage"])
    detection.metadata = metadata

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = box_annotator.annotate(scene=image, detections=detection)
    labels = [f"{metadata.thing_classes[cid]} {conf*100:.2f}%" for cid, conf in zip(detection.class_id, detection.confidence)]
    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=detection,
        labels=labels
    )
    
    cn = [metadata.thing_classes[cid] for cid in detection.class_id]
    con = [conf for conf in detection.confidence]

    print(Counter(cn).keys())
    print(Counter(cn).values())

    print("average confidence : %.2f%", (sum(con)/len(con)*100))

    return annotated_image, len(cn), sum(con)/len(con)*100
  except Exception as e:
    print(f"[ERROR in partDetectionCallback] {e}")
    return image