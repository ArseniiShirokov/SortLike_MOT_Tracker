#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import json
import math
import os

import subprocess
import numpy
from munkres import Munkres, make_cost_matrix
from tqdm import tqdm

from dataset_db import dataset_db
from utils import Bbox, filter_detections, get_frame_id_delta

frames_iterable = None

def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Visual tracking and '
                    'hungarian matching by IOU metric'
    )

    parser.add_argument(
        '-i',
        '--dataset-name-or-path',
        type=str,
        help='Dataset name or path',
        required=True
    )

    parser.add_argument(
        '-p',
        '--parameters',
        dest='parameters_file',
        type=argparse.FileType('r'),
        help='Parameters file',
        required=True
    )

    parser.add_argument(
        '-d',
        '--detections',
        dest='detections_file',
        type=argparse.FileType('r'),
        help='Detections file',
        required=True
    )

    parser.add_argument(
        '-r',
        '--results',
        dest='results_file',
        type=argparse.FileType('w'),
        help='Results file',
        required=True
    )

    parser.add_argument(
        '--indent',
        type=int,
        help='Sets indent for the output json'
    )

    parser.add_argument(
        '-q', '--quite',
        action='store_true',
        default=False,
        help='Set this to hide output'
    )

    return parser.parse_args()


class VOT(object):
    def __init__(
            self,
            dataset_path,
            tracking_plugin_docker_image,
            first_frame_id=0
    ):
        self.asms_plugin_proc = subprocess.Popen(
            f"docker run --rm -i -v $(pwd)/{dataset_path}:/dataset "
            f"{tracking_plugin_docker_image}",
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )
        self.src_image_path = generate_docker_path(first_frame_id) 
        
        
    def to_next_frame(self, dst_frame_id, first=False):
        if (not first):
            self.asms_plugin_proc.stdin.write(
               '0 0 0 0'
            )
            self.asms_plugin_proc.stdin.flush()
        
        dst_image_path = generate_docker_path(dst_frame_id) 
        self.asms_plugin_proc.stdin.write(f'{self.src_image_path}\n{dst_image_path}\n')
        self.asms_plugin_proc.stdin.flush()
        self.src_image_path = dst_image_path
    
    
    def get_estimate(self, target_bbox):
        ltwh = target_bbox.ltwh
        self.asms_plugin_proc.stdin.write(
           '{0} {1} {2} {3}\n'.format(ltwh[0], ltwh[1], ltwh[2], ltwh[3])
        )
        self.asms_plugin_proc.stdin.flush()
        res = self.asms_plugin_proc.stdout.readline()
        
        dst_bbox = list(map(int, res.split()[:4]))
        return Bbox('ltwh', dst_bbox[0], dst_bbox[1], dst_bbox[2], dst_bbox[3])
    
    
    def finish(self):
        subprocess.run('docker rm -f $(docker ps -a -q)', shell=True, check=True)



def generate_docker_path(frame_id):
    frame = frames_iterable[frame_id]
    return os.path.join(
            '/dataset',
            frame['file']
        )


class Track(object):
    total_tracks = 0

    def __init__(
            self,
            frame_id,
            detection
    ):
        self.id = Track.total_tracks
        Track.total_tracks += 1
        self.detections_frames = []
        self.detections = []
        
        self.cur_pos = detection
        self.cur_fr_id = frame_id
        
        self.buffered_detections = []
        self.buffered_detections_frames = []
        self.add_detection(frame_id, detection) 
    
        

    def add_detection(self, frame_id, det):
        self.detections += self.buffered_detections[:-1]
        self.detections_frames += self.buffered_detections_frames[:-1]
        self.buffered_detections = []
        self.buffered_detections_frames = []
        self.detections_frames.append(frame_id)
        self.detections.append(det)
        
        self.cur_pos = det
        self.cur_fr_id = frame_id
        

    def predict(self, visual_tracker, new_fr_id): 
        target_bbox = self.cur_pos
        new_pos = visual_tracker.get_estimate( 
            target_bbox
        )
        
        self.cur_pos = new_pos
        self.cur_fr_id = new_fr_id
        
        self.buffered_detections.append(self.cur_pos)
        self.buffered_detections_frames.append(self.cur_fr_id) 
        
        
    def get_estimate(self):
        return self.cur_pos
        
        

def main():
    args = parse_arguments()

    descr = \
    dataset_db.load_dataset_descr_by_name_or_path(args.dataset_name_or_path)
    fps = descr['fps']
    frame_secs = 1.0 / fps

    dataset_path = dataset_db.get_path_by_name_or_path(args.dataset_name_or_path)

    parameters = json.load(args.parameters_file)

    frames_per_key_frame = max(
        1,
        int(math.ceil(
            parameters['min_seconds_between_detections'] / frame_secs
        ))
    )
    frames_per_vt_frame = max(
        1,
        int(math.ceil(
            parameters['min_seconds_between_vt'] / frame_secs
        ))
    )

    detections = filter_detections(
        json.load(args.detections_file),
        object_type=parameters['object_type'],
        detection_type=parameters['detection_type']
    )
    assert detections['dataset_name'] == descr['name']

    results = {
        'dataset_name': descr['name'],
        'algorithm': 'asms_baseline',
        'parameters': parameters,
        'frames': [{'file': x['file']} for x in detections['frames']]
    }

    tracks = []
    result_tracks = []
    
    global frames_iterable
    frames_iterable = detections['frames']
    
    if not args.quite:
        frames_iterable = tqdm(
            frames_iterable,
            desc='Simple ASMS tracking',
            dynamic_ncols=True
        )
    # Start VT
    visual_tracker = VOT(dataset_path, parameters['tracking_plugin_docker_image'], first_frame_id=0)

    for frame_id, det_frame_info in enumerate(frames_iterable):
        # Don't take vt too often
        if frame_id % frames_per_key_frame != 0 and frame_id % frames_per_vt_frame != 0:
            continue
        
        # Make asms estimates
        visual_tracker.to_next_frame(frame_id, frame_id == 0)
        for t in tracks:
            t.predict(visual_tracker, frame_id)
        
        # Don't take detections too often
        if frame_id % frames_per_key_frame != 0:
            continue

        # Skip frames with no detections
        if det_frame_info.get('objects') is None:
            continue

        # We provide results on this frame
        results['frames'][frame_id]['objects'] = []

        # Filter detections
        dets = []
        for obj in det_frame_info.get('objects'):
            det = obj['detection']

            if 'confidence' in det and det['confidence'] \
                    < parameters['detection_confidence_threshold']:
                continue

            det['bbox'] = Bbox.from_json(det['bbox'])
            dets.append(det)

        # Get matching distances
        distances = numpy.zeros((len(tracks), len(dets)))
        for i, track in enumerate(tracks):
            track_bbox = track.get_estimate() 
            for j, det in enumerate(dets):
                distances[i, j] = Bbox.iou(track_bbox, det['bbox'])

        # Perform matching
        if len(tracks) == 0 or len(dets) == 0:
            indices = []
        else:
            distances[distances < parameters['iou_threshold']] = 0
            # distances is `profit` matrix so converting it to `cost`
            indices = Munkres().compute(make_cost_matrix(distances))

        matched_tracks = set()
        matched_detections = set()
        for i, j in indices:
            iou = distances[i, j]

            if iou < parameters['iou_threshold']:
                continue

            if i in matched_tracks:
                continue

            if j in matched_detections:
                continue

            tracks[i].add_detection(
                frame_id,
                dets[j]['bbox']
            )
            matched_tracks.add(i)
            matched_detections.add(j)

        # Deal with not matched tracks
        filtered_tracks = []
        for i, track in enumerate(tracks):
            if i in matched_tracks:
                filtered_tracks.append(track)
                continue

            if (frame_id - track.detections_frames[-1]) * frame_secs \
                    > parameters['max_seconds_not_tracked']:
                # Can't continue this track
                result_tracks.append(track)
            else:
                filtered_tracks.append(track)
        tracks = filtered_tracks

        # Deal with unmatched detections
        for j in set(range(len(dets))) - matched_detections:
            tracks.append(Track(
                frame_id,
                dets[j]['bbox']
            ))

    visual_tracker.finish()
    result_tracks += tracks

    # Filter by min_track_detections
    result_tracks = [
        x for x in result_tracks
        if len(x.detections) >= parameters['min_track_detections']
    ]

    for track in result_tracks:
        for frame_id, det in zip(track.detections_frames, track.detections):
            results['frames'][frame_id].setdefault('objects', [])
            objects = results['frames'][frame_id]['objects']
            objects.append({
                'type': parameters['object_type'],
                'track_id': track.id,
                'detections': [
                    {
                        'bbox': det.json(),
                        'type': parameters['detection_type']
                    }
                ]
            })

    json.dump(results, args.results_file, indent=args.indent)


if __name__ == '__main__':
    main()
