import _thread as thread
from tf_detector import TfDetector
from tracker import Tracker
from setting import *
from datetime import datetime
import numpy as np
import func
import time
import requests
import base64
import ast
import cv2
import sys


class WorkCamera:

    def __init__(self, camera_list):
        self.class_model = TfDetector()

        self.camera_list = camera_list
        self.cap_list = []
        self.video_writer_list = []
        self.frame_list = []
        self.update_frame = []
        self.buffer_frame_list = []
        self.buffer_tracker_images = []
        self.buffer_tracker_rect = []
        self.buffer_tracker_distance = []
        self.ret_image = []
        self.detect_rects_list = []
        self.frame_ind_list = []
        self.tracker_list = []
        self.camera_enable = []
        self.camera_size_list = []
        self.camera_roi_list = []
        self.total_in_list = []
        self.total_out_list = []
        self.obj_ind_list = []
        self.direction_list = []
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')

        for i in range(len(camera_list)):
            if camera_list[i] == '' or camera_list[i] is None:
                self.cap_list.append(None)
                self.camera_size_list.append(None)
                self.camera_roi_list.append(None)
                self.video_writer_list.append(None)
            else:
                cap = cv2.VideoCapture(camera_list[i])
                self.cap_list.append(cap)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = cap.get(cv2.CAP_PROP_FPS)
                self.camera_size_list.append([width, height])
                nw = int(width * RESIZE_FACTOR)
                nh = int(height * RESIZE_FACTOR)
                roi = CAMERA_ROI[i]
                x1, y1, x2, y2 = int(roi[0] * nw), int(roi[1] * nh), int(roi[2] * nw), int(roi[3] * nh)
                self.camera_roi_list.append([x1, y1, x2, y2])
                if SAVE_CROP:
                    self.video_writer_size = (x2 - x1 + 40, y2 - y1 + 70)
                else:
                    self.video_writer_size = (nw, nh)

                self.video_writer_list.append(
                    cv2.VideoWriter('result{}.avi'.format(i), self.fourcc, self.fps, self.video_writer_size))

            self.tracker_list.append(Tracker())
            self.frame_list.append(None)
            self.update_frame.append(False)
            self.ret_image.append(None)
            self.detect_rects_list.append([])
            self.obj_ind_list.append(None)
            self.frame_ind_list.append(0)
            self.total_in_list.append(0)
            self.total_out_list.append(0)
            self.camera_enable.append(True)
            self.buffer_frame_list.append([])

            line_p1 = np.array([CHECK_LINE[i][0], CHECK_LINE[i][1]])
            line_p2 = np.array([CHECK_LINE[i][2], CHECK_LINE[i][3]])
            p3 = np.array(DIRECTION[i]['start'])
            p4 = np.array(DIRECTION[i]['end'])
            d1 = np.cross(line_p2 - line_p1, p3 - line_p1) / np.linalg.norm(line_p2 - line_p1)
            d2 = np.cross(line_p2 - line_p1, p4 - line_p1) / np.linalg.norm(line_p2 - line_p1)
            if d1 < 0 and d2 > 0:
                self.direction_list.append(True)
            else:
                self.direction_list.append(False)

    def read_frame(self, camera_ind, scale_factor=1.0):
        while True:
            if self.cap_list[camera_ind] is None:
                self.frame_list[camera_ind] = None

            elif self.camera_enable[camera_ind]:
                ret, frame = self.cap_list[camera_ind].read()

                if ret:
                    if scale_factor != 1.0:
                        frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
                    self.frame_list[camera_ind] = frame
                    self.buffer_frame_list[camera_ind].append(frame)
                    self.update_frame[camera_ind] = True
                else:
                    cam_url = self.camera_list[camera_ind]
                    print("Fail to read camera!", cam_url)
                    self.cap_list[camera_ind].release()
                    time.sleep(0.5)
                    self.cap_list[camera_ind] = cv2.VideoCapture(cam_url)

            time.sleep(0.08)

    def process_frame(self, cam_ind):
        while True:
            if self.update_frame[cam_ind]:
                if self.frame_list[cam_ind] is not None:
                    self.frame_ind_list[cam_ind] += 1

                    buffer_size = len(self.buffer_frame_list[cam_ind])
                    print(buffer_size)
                    img_list = []
                    if DETECT_GROUP and buffer_size >= 8:
                        for i in range(8):
                            img_list.append(self.buffer_frame_list[cam_ind][int(buffer_size / 8 * (i + 1)) - 1])
                    elif DETECT_GROUP and buffer_size >= 4:
                        for i in range(4):
                            img_list.append(self.buffer_frame_list[cam_ind][int(buffer_size / 4 * (i + 1)) - 1])
                    else:
                        img_list = [self.frame_list[cam_ind].copy()]

                    self.buffer_frame_list[cam_ind] = []

                    if DETECT_ENABLE:
                        det_rect_list, det_score_list, det_class_list = self.detect_frame(img_list, cam_ind)

                        for i in range(len(det_rect_list)):
                            valid_rects, _, valid_distance = \
                                self.check_valid_detection(det_rect_list[i], det_score_list[i], det_class_list[i], cam_ind)

                            self.buffer_tracker_images.append(img_list[i])
                            self.buffer_tracker_rect.append(valid_rects)
                            self.buffer_tracker_distance.append(valid_distance)

                    else:
                        self.buffer_tracker_images.append(img_list[0])
                        self.buffer_tracker_rect.append([])
                        self.buffer_tracker_distance.append([])

                # initialize the variable
                self.update_frame[cam_ind] = False

            time.sleep(0.01)

    def process_tracker(self, cam_ind):
        prev_time = 0
        while True:
            if len(self.buffer_tracker_rect) > 0:
                img_color = self.buffer_tracker_images[0]
                valid_rects = self.buffer_tracker_rect[0]
                valid_distance = self.buffer_tracker_distance[0]

                cnt_in, cnt_out = self.tracker_list[0].update(valid_rects,
                                                              valid_distance)
                if self.direction_list[cam_ind]:
                    self.total_in_list[cam_ind] += cnt_in
                    self.total_out_list[cam_ind] += cnt_out
                else:
                    self.total_in_list[cam_ind] += cnt_out
                    self.total_out_list[cam_ind] += cnt_in

                if DISPLAY_TRACK_INDEX:
                    tracker_rect_list = []
                    ind_list = []
                    for tracker_ind in self.tracker_list[cam_ind].items.keys():
                        tracker_rect_list.append(self.tracker_list[cam_ind].items[tracker_ind]['rect'][-1])
                        ind_list.append(tracker_ind)
                    img_ret = self.draw_image(img_color, tracker_rect_list, cam_ind, ind_list)
                    self.detect_rects_list[cam_ind] = tracker_rect_list
                    self.obj_ind_list[cam_ind] = ind_list
                else:
                    img_ret = self.draw_image(img_color, valid_rects, cam_ind)
                    self.detect_rects_list[cam_ind] = valid_rects
                    self.obj_ind_list[cam_ind] = None

                if DISPLAY_DETECT_FRAME_ONLY:
                    self.ret_image[cam_ind] = img_ret
                    if SAVE_VIDEO:
                        if int(time.time() / SAVE_PERIOD) != int(prev_time / SAVE_PERIOD):
                            prev_time = time.time()
                            self.video_writer_list[cam_ind].release()
                            self.video_writer_list[cam_ind] = cv2.VideoWriter('result_{}.avi'.format(int(time.time() / SAVE_PERIOD)), self.fourcc, self.fps, self.video_writer_size)

                        [x1, y1, x2, y2] = self.camera_roi_list[cam_ind]
                        self.video_writer_list[cam_ind].write(img_ret[y1-50:y2+20, x1-20:x2+20])

                self.buffer_tracker_images.pop(0)
                self.buffer_tracker_distance.pop(0)
                self.buffer_tracker_rect.pop(0)

            time.sleep(0.08)

    def check_valid_detection(self, rect_list, score_list, class_list, cam_ind):
        check_rect_list = []
        check_score_list = []

        # ----------------------- check validation using threshold and ROI ------------------------
        for i in range(len(score_list)):
            if class_list[i] != 1 or score_list[i] < DETECTION_THRESHOLD:
                continue

            # check ROI
            if rect_list[i][0] + rect_list[i][2] < self.camera_roi_list[cam_ind][0] * 2:
                continue
            elif rect_list[i][0] + rect_list[i][2] > self.camera_roi_list[cam_ind][2] * 2:
                continue
            elif rect_list[i][1] + rect_list[i][3] < self.camera_roi_list[cam_ind][1] * 2:
                continue
            elif rect_list[i][1] + rect_list[i][3] > self.camera_roi_list[cam_ind][3] * 2:
                continue

            # check overlap with other rects
            f_overlap = False
            for j in range(len(check_rect_list)):
                if func.check_overlap_rect(check_rect_list[j], rect_list[i]):
                    if check_score_list[j] < score_list[i]:
                        check_score_list[j] = score_list[i]
                        check_rect_list[j] = rect_list[i]
                    f_overlap = True
                    break

            if f_overlap:
                continue

            # check width/height rate
            w = rect_list[i][2] - rect_list[i][0]
            h = rect_list[i][3] - rect_list[i][1]
            if max(w, h) / min(w, h) > 2:
                continue
            elif max(w, h) > SIZE_THRESHOLD:
                continue

            # register data
            check_rect_list.append(rect_list[i])
            check_score_list.append(score_list[i])

        # ------------ check validation by check body and calculate distance from cross line ------
        final_rect = []
        final_score = []
        final_distance = []

        frame_height, frame_width = self.frame_list[cam_ind].shape[:2]
        line_p1 = np.array([int(CHECK_LINE[cam_ind][0] * frame_width), int(CHECK_LINE[cam_ind][1] * frame_height)])
        line_p2 = np.array([int(CHECK_LINE[cam_ind][2] * frame_width), int(CHECK_LINE[cam_ind][3] * frame_height)])

        for i in range(len(check_rect_list)):
            r_body = check_rect_list[i]
            body_h = r_body[3] - r_body[1]
            body_w = r_body[2] - r_body[0]
            f_ignore = False
            for j in range(len(check_rect_list)):
                r2 = check_rect_list[j]
                if r_body[0] < (r2[0] + r2[2]) / 2 < r_body[2] or r2[0] < (r_body[0] + r_body[2]) / 2 < r2[2]:
                    if abs(r_body[1] - r2[3]) * 1.5 < r2[3] - r2[1] and r_body[3] > r2[3] + 0.5 * (r2[3] - r2[1]):
                        if body_h > 1.2 * (r2[3] - r2[1]) or body_w > 1.2 * (r2[2] - r2[0]):
                            f_ignore = True
                            break

            if not f_ignore:
                final_rect.append(check_rect_list[i])
                final_score.append(check_score_list[i])

                p3 = np.array([(check_rect_list[i][0] + check_rect_list[i][2]) / 2,
                               (check_rect_list[i][1] + check_rect_list[i][3]) / 2])
                final_distance.append(np.cross(line_p2-line_p1, p3-line_p1)/np.linalg.norm(line_p2-line_p1))

        return final_rect, final_score, final_distance

    def draw_image(self, img, rects, cam_ind, obj_ind=None):
        # draw ROI region
        h, w = self.frame_list[cam_ind].shape[:2]
        roi_pt1 = (int(CAMERA_ROI[cam_ind][0] * w), int(CAMERA_ROI[cam_ind][1] * h))
        roi_pt2 = (int(CAMERA_ROI[cam_ind][2] * w), int(CAMERA_ROI[cam_ind][3] * h))
        cv2.rectangle(img, roi_pt1, roi_pt2, (100, 100, 100), 1)

        # draw check line
        line_pt1 = (int(CHECK_LINE[cam_ind][0] * w), int(CHECK_LINE[cam_ind][1] * h))
        line_pt2 = (int(CHECK_LINE[cam_ind][2] * w), int(CHECK_LINE[cam_ind][3] * h))
        cv2.line(img, line_pt1, line_pt2, (0, 0, 255), 2)

        # draw direction arrow
        direct_start = (int(DIRECTION[cam_ind]['start'][0] * w), int(DIRECTION[cam_ind]['start'][1] * h))
        direct_end = (int(DIRECTION[cam_ind]['end'][0] * w), int(DIRECTION[cam_ind]['end'][1] * h))
        cv2.circle(img, direct_start, 8, (255, 0, 0), -1)
        cv2.line(img, direct_start, direct_end, (0, 255, 0), 2)

        # draw objects with rectangle
        for i in range(len(rects)):
            rect = rects[i]
            cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 2)
            cv2.putText(img, str(max(rect[2] - rect[0], rect[3] - rect[1])), (rect[0], rect[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if obj_ind is not None:
                cv2.putText(img, str(obj_ind[i]), (rect[0], rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # write counting result
        if SAVE_CROP:
            x, y = 320, 70
            cv2.rectangle(img, (x, y), (x + 360, y + 30), (255, 255, 255), -1)
            str_in = "{}, In: {}, Out: {}".format(datetime.now().strftime("%H:%M:%S"), self.total_in_list[cam_ind],
                                                  self.total_out_list[cam_ind])
            cv2.putText(img, str_in, (x + 10, y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        else:
            x, y = 0, 20
            cv2.rectangle(img, (x, y), (x + 240, y + 70), (255, 255, 255), -1)
            str_in = "Incoming: {}".format(self.total_in_list[cam_ind])
            str_out = "Outgoing: {}".format(self.total_out_list[cam_ind])
            cv2.putText(img, str_in, (x + 10, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(img, str_out, (x + 10, y + 58), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return img

    @staticmethod
    def detect_service(img):
        save_file = str(time.time()) + '.jpg'
        cv2.imwrite(save_file, img)
        file_data = open(save_file, 'rb')

        json_data = {
            "action": 'detect',
            "image": base64.b64encode(file_data.read()).decode('UTF-8')
        }
        response = requests.post(url=SERVICE_URL, json=json_data)

        response_json = response.json()
        if 'result' in response_json and response_json['result']['state']:
            rects = ast.literal_eval(response_json['result']['rect'])
            scores = ast.literal_eval(response_json['result']['score'])
            classes = ast.literal_eval(response_json['result']['class'])
        else:
            rects, scores, classes = [], [], []

        file_data.close()
        func.rm_file(save_file)

        return rects, scores, classes

    def detect_frame(self, img_list, cam_ind):
        roi = [max(0, self.camera_roi_list[cam_ind][0] - 20),
               max(0, self.camera_roi_list[cam_ind][1] - 20),
               min(self.camera_size_list[cam_ind][0], self.camera_roi_list[cam_ind][2] + 20),
               min(self.camera_size_list[cam_ind][1], self.camera_roi_list[cam_ind][3] + 20)]

        w = roi[2] - roi[0]
        h = roi[3] - roi[1]

        crop_list = []
        for i in range(len(img_list)):
            crop_list.append(img_list[i][roi[1]:roi[3], roi[0]:roi[2]])

        if len(crop_list) == 8:
            img_crop1 = np.concatenate((np.concatenate((crop_list[0], crop_list[1]), axis=0),
                                        np.concatenate((crop_list[2], crop_list[3]), axis=0)), axis=1)
            img_crop2 = np.concatenate((np.concatenate((crop_list[4], crop_list[5]), axis=0),
                                        np.concatenate((crop_list[6], crop_list[7]), axis=0)), axis=1)
            img_crop = np.concatenate((img_crop1, img_crop2), axis=0)
        elif len(crop_list) == 4:
            img_crop = np.concatenate((np.concatenate((crop_list[0], crop_list[1]), axis=0),
                                       np.concatenate((crop_list[2], crop_list[3]), axis=0)), axis=1)
        else:
            img_crop = crop_list[0]

        if USING_SERVICE:
            det_rect_list, det_score_list, det_class_list = self.detect_service(img_crop)
        else:
            det_rect_list, det_score_list, det_class_list = self.class_model.detect_from_images(img_crop)

        if len(crop_list) == 8:
            rect_list = [[], [], [], [], [], [], [], []]
            score_list = [[], [], [], [], [], [], [], []]
            class_list = [[], [], [], [], [], [], [], []]

            for i in range(len(det_rect_list)):
                [x1, y1, x2, y2] = det_rect_list[i]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if cx < w:
                    if cy < h:
                        rect_list[0].append([x1 + roi[0], y1 + roi[1], x2 + roi[0], y2 + roi[1]])
                        score_list[0].append(det_score_list[i])
                        class_list[0].append(det_class_list[i])
                    elif cy < 2 * h:
                        rect_list[1].append([x1 + roi[0], y1 - h + roi[1], x2 + roi[0], y2 - h + roi[1]])
                        score_list[1].append(det_score_list[i])
                        class_list[1].append(det_class_list[i])
                    elif cy < 3 * h:
                        rect_list[4].append([x1 + roi[0], y1 - 2 * h + roi[1], x2 + roi[0], y2 - 2 * h + roi[1]])
                        score_list[4].append(det_score_list[i])
                        class_list[4].append(det_class_list[i])
                    else:
                        rect_list[5].append([x1 + roi[0], y1 - 3 * h + roi[1], x2 + roi[0], y2 - 3 * h + roi[1]])
                        score_list[5].append(det_score_list[i])
                        class_list[5].append(det_class_list[i])
                else:
                    if cy < h:
                        rect_list[2].append([x1 - w + roi[0], y1 + roi[1], x2 - w + roi[0], y2 + roi[1]])
                        score_list[2].append(det_score_list[i])
                        class_list[2].append(det_class_list[i])
                    elif cy < 2 * h:
                        rect_list[3].append([x1 - w + roi[0], y1 - h + roi[1], x2 - w + roi[0], y2 - h + roi[1]])
                        score_list[3].append(det_score_list[i])
                        class_list[3].append(det_class_list[i])
                    elif cy < 3 * h:
                        rect_list[6].append([x1 - w + roi[0], y1 - 2 * h + roi[1], x2 - w + roi[0], y2 - 2 * h + roi[1]])
                        score_list[6].append(det_score_list[i])
                        class_list[6].append(det_class_list[i])
                    else:
                        rect_list[7].append([x1 - w + roi[0], y1 - 3 * h + roi[1], x2 - w + roi[0], y2 - 3 * h + roi[1]])
                        score_list[7].append(det_score_list[i])
                        class_list[7].append(det_class_list[i])

        elif len(crop_list) == 4:
            rect_list = [[], [], [], []]
            score_list = [[], [], [], []]
            class_list = [[], [], [], []]

            for i in range(len(det_rect_list)):
                [x1, y1, x2, y2] = det_rect_list[i]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if cx < w and cy < h:
                    rect_list[0].append([x1 + roi[0], y1 + roi[1], x2 + roi[0], y2 + roi[1]])
                    score_list[0].append(det_score_list[i])
                    class_list[0].append(det_class_list[i])
                elif cx < w and cy >= h:
                    rect_list[1].append([x1 + roi[0], y1 - h + roi[1], x2 + roi[0], y2 - h + roi[1]])
                    score_list[1].append(det_score_list[i])
                    class_list[1].append(det_class_list[i])
                elif cx >= w and cy < h:
                    rect_list[2].append([x1 - w + roi[0], y1 + roi[1], x2 - w + roi[0], y2 + roi[1]])
                    score_list[2].append(det_score_list[i])
                    class_list[2].append(det_class_list[i])
                else:
                    rect_list[3].append([x1 - w + roi[0], y1 - h + roi[1], x2 - w + roi[0], y2 - h + roi[1]])
                    score_list[3].append(det_score_list[i])
                    class_list[3].append(det_class_list[i])

        else:
            score_list = [det_score_list]
            class_list = [det_class_list]
            rect_list = [[]]

            for i in range(len(det_rect_list)):
                [x1, y1, x2, y2] = det_rect_list[i]
                rect_list[0].append([x1 + roi[0], y1 + roi[1], x2 + roi[0], y2 + roi[1]])

        return rect_list, score_list, class_list

    def run_thread(self):
        for i in range(len(self.cap_list)):
            thread.start_new_thread(self.read_frame, (i, RESIZE_FACTOR))
            thread.start_new_thread(self.process_frame, (i, ))
            thread.start_new_thread(self.process_tracker, (i,))

        while True:
            if SHOW_VIDEO:
                for i in range(len(self.cap_list)):
                    if DISPLAY_DETECT_FRAME_ONLY:
                        if self.ret_image[i] is not None:
                            cv2.imshow('org' + str(i), self.ret_image[i])
                    else:
                        if self.frame_list[i] is not None:
                            img_org = self.draw_image(self.frame_list[i].copy(),
                                                      rects=self.detect_rects_list[i],
                                                      cam_ind=i,
                                                      obj_ind=self.obj_ind_list[i])
                            cv2.imshow('org' + str(i), img_org)

            key = cv2.waitKey(10)
            if key == ord('q'):
                break

    def run(self):
        frame_ind = 0
        key_time = 1
        while True:
            frame_ind += 1
            ret, frame = self.cap_list[0].read()
            if not ret:
                break

            # resize image
            if RESIZE_FACTOR != 1.0:
                frame = cv2.resize(frame, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

            self.frame_list[0] = frame

            # detect
            if DETECT_ENABLE:
                t1 = time.time()
                det_rect_list, det_score_list, det_class_list = self.detect_frame([frame], 0)
                valid_rects, _, valid_distance = \
                    self.check_valid_detection(det_rect_list[0], det_score_list[0], det_class_list[0], 0)
                print(time.time() - t1)
            else:
                valid_rects = []
                valid_distance = []

            cnt_in, cnt_out = self.tracker_list[0].update(valid_rects, valid_distance)
            if self.direction_list[0]:
                self.total_in_list[0] += cnt_in
                self.total_out_list[0] += cnt_out
            else:
                self.total_in_list[0] += cnt_out
                self.total_out_list[0] += cnt_in

            if DISPLAY_TRACK_INDEX:
                tracker_rect_list = []
                ind_list = []
                for tracker_ind in self.tracker_list[0].items.keys():
                    tracker_rect_list.append(self.tracker_list[0].items[tracker_ind]['rect'][-1])
                    ind_list.append(tracker_ind)
                img_ret = self.draw_image(frame, tracker_rect_list, 0, ind_list)
            else:
                img_ret = self.draw_image(frame, valid_rects, 0)

            if SHOW_VIDEO:
                cv2.imshow('ret', img_ret)

            if SAVE_VIDEO:
                self.video_writer_list[0].write(img_ret)

            key = cv2.waitKey(key_time)
            if key == ord('q'):
                break
            elif key == ord('+'):
                key_time = 1
            elif key == ord('-'):
                key_time = 500
            elif key == ord('s'):
                key_time = 0


if __name__ == '__main__':
    if len(sys.argv) > 1:
        cam_list = [sys.argv[1]]
    else:
        cam_list = CAMERA_URL

    class_work = WorkCamera(cam_list)

    if RUN_MODE_THREAD:
        class_work.run_thread()
    else:
        class_work.run()
