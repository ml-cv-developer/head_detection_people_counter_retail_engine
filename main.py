import _thread as thread
from tf_detector import TfDetector
from tracker import Tracker
from setting import *
import numpy as np
import func
import time
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
        self.ret_image = []
        self.detect_rects_list = []
        self.frame_ind_list = []
        self.tracker_list = []
        self.camera_list = camera_list
        self.camera_enable = []
        self.camera_size_list = []
        self.camera_roi_list = []
        self.total_in_list = []
        self.total_out_list = []
        self.obj_ind_list = []
        self.direction_list = []
        fourcc = cv2.VideoWriter_fourcc(*'MPEG')

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
                fps = cap.get(cv2.CAP_PROP_FPS)
                self.camera_size_list.append([width, height])
                nw = int(width * RESIZE_FACTOR)
                nh = int(height * RESIZE_FACTOR)
                roi = CAMERA_ROI[i]
                self.camera_roi_list.append([int(roi[0] * nw), int(roi[1] * nh), int(roi[2] * nw), int(roi[3] * nh)])
                self.video_writer_list.append(cv2.VideoWriter('result{}.avi'.format(i), fourcc, fps, (nw, nh)))

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
                    self.update_frame[camera_ind] = True
                else:
                    cam_url = self.camera_list[camera_ind]
                    print("Fail to read camera!", cam_url)
                    self.cap_list[camera_ind].release()
                    time.sleep(0.5)
                    self.cap_list[camera_ind] = cv2.VideoCapture(cam_url)

            time.sleep(0.05)

    def process_frame(self, cam_ind):
        while True:
            if self.update_frame[cam_ind]:
                if self.frame_list[cam_ind] is not None:
                    self.frame_ind_list[cam_ind] += 1

                    img_color = self.frame_list[cam_ind].copy()

                    if DETECT_ENABLE:
                        det_rect_list, det_score_list, det_class_list = self.detect_frame(img_color, cam_ind)
                        valid_rects, _, valid_distance = \
                            self.check_valid_detection(det_rect_list, det_score_list, det_class_list, cam_ind)
                    else:
                        valid_rects = []
                        valid_distance = []

                    # tracker
                    cnt_in, cnt_out = self.tracker_list[0].update(self.frame_ind_list[cam_ind],
                                                                  valid_rects,
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

                # initialize the variable
                self.update_frame[cam_ind] = False

            time.sleep(0.1)

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
            if obj_ind is not None:
                cv2.putText(img, str(obj_ind[i]), (rect[0], rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # write counting result
        cv2.rectangle(img, (0, 0), (240, 70), (255, 255, 255), -1)
        str_in = "Incoming: {}".format(self.total_in_list[cam_ind])
        str_out = "Outgoing: {}".format(self.total_out_list[cam_ind])
        cv2.putText(img, str_in, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img, str_out, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return img

    def detect_frame(self, img, cam_ind):
        roi = [max(0, self.camera_roi_list[cam_ind][0] - 20),
               max(0, self.camera_roi_list[cam_ind][1] - 20),
               min(self.camera_size_list[cam_ind][0], self.camera_roi_list[cam_ind][2] + 20),
               min(self.camera_size_list[cam_ind][1], self.camera_roi_list[cam_ind][3] + 20)]

        img_crop = img[roi[1]:roi[3], roi[0]:roi[2]]
        det_rect_list, det_score_list, det_class_list = self.class_model.detect_from_images(img_crop)
        for i in range(len(det_rect_list)):
            det_rect_list[i][0] += roi[0]
            det_rect_list[i][1] += roi[1]
            det_rect_list[i][2] += roi[0]
            det_rect_list[i][3] += roi[1]

        return det_rect_list, det_score_list, det_class_list

    def run_thread(self):
        for i in range(len(self.cap_list)):
            thread.start_new_thread(self.read_frame, (i, RESIZE_FACTOR))
            thread.start_new_thread(self.process_frame, (i, ))

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
                det_rect_list, det_score_list, det_class_list = self.detect_frame(frame, 0)
                valid_rects, _, valid_distance = \
                    self.check_valid_detection(det_rect_list, det_score_list, det_class_list, 0)
                print(time.time() - t1)
            else:
                valid_rects = []
                valid_distance = []

            cnt_in, cnt_out = self.tracker_list[0].update(frame_ind, valid_rects, valid_distance)
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
