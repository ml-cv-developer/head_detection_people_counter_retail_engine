from setting import *
import func
import numpy as np
import time


class Tracker:
    def __init__(self):
        self.items = {}
        self.total = 0

    def __get_closest_pair(self, rect_list, cnt):
        """
            Compare the tracker data and rect_list, and then return the pair of the closest points
            - return format:
                    [[0, 2, 3, ..., n],     # index of rect_list
                     [1, 2, 4, ..., n]]     # index of tracker
        """
        list_pair = []
        list_dist = []
        list_pair_update = [[], []]

        for i in range(len(rect_list)):
            for tracker_ind in self.items:
                d = func.get_distance_rect(rect_list[i], self.items[tracker_ind]['rect'][-1], x_rate=9)
                if len(self.items[tracker_ind]['rect']) <= 2 and cnt > self.items[tracker_ind]['cnt'] + 1:
                    d = d + (cnt - self.items[tracker_ind]['cnt']) * 15
                if d < TRACKER_THRESHOLD_DISTANCE:
                    list_pair.append([i, tracker_ind])
                    list_dist.append(d)

        while True:
            if not list_dist:
                break

            min_ind = int(np.argmin(list_dist))
            min_pair = list_pair[min_ind]

            list_pair.pop(min_ind)
            list_dist.pop(min_ind)

            f_ignore = False
            for i in range(len(list_pair_update[0])):
                if list_pair_update[0][i] == min_pair[0] or list_pair_update[1][i] == min_pair[1]:
                    f_ignore = True
                    break

            if not f_ignore:
                list_pair_update[0].append(min_pair[0])
                list_pair_update[1].append(min_pair[1])

            if not list_pair:
                break

        return list_pair_update

    def __update_tracker(self, tracker_ind, cnt, rect, distance):
        """
            Update the tracker_ind item with new data
        """
        self.items[tracker_ind]['cnt'] = cnt
        self.items[tracker_ind]['ts'].append(time.time())
        self.items[tracker_ind]['rect'].append(rect)
        self.items[tracker_ind]['distance'].append(distance)
        cnt_in = 0
        cnt_out = 0

        if distance > DISTANCE_MARGIN_IN and self.items[tracker_ind]['status'] != 'in':
            f_check = False
            for i in range(len(self.items[tracker_ind]['distance'])):
                prev_pos = self.items[tracker_ind]['distance'][i]
                if distance - prev_pos > DISTANCE_THRESHOLD:
                    if (prev_pos < 20 and i < 2) or prev_pos < DISTANCE_MARGIN_OUT:
                        f_check = True
                        break

            if f_check:
                self.items[tracker_ind]['status'] = 'in'
                cnt_in = 1

        if distance < DISTANCE_MARGIN_OUT and self.items[tracker_ind]['status'] != 'out':
            f_check = False
            for i in range(len(self.items[tracker_ind]['distance'])):
                prev_pos = self.items[tracker_ind]['distance'][i]
                if prev_pos - distance > DISTANCE_THRESHOLD and prev_pos > DISTANCE_MARGIN_IN:
                    f_check = True
                    break

            if f_check:
                self.items[tracker_ind]['status'] = 'out'
                cnt_out = 1

        return cnt_in, cnt_out

    def __add_tracker(self, cnt, rect, distance):
        """
            Add new tracker with new data
        """
        self.items[self.total] = {'cnt': cnt,
                                  'ts': [time.time()],
                                  'rect': [rect],
                                  'distance': [distance],
                                  'status': ''}

    def __delete_tracker(self, cnt):
        temp_tracker = {}
        for tracker_ind in self.items:
            if self.items[tracker_ind]['cnt'] + TRACKER_KEEP_LENGTH >= cnt:
                len_tracker_item = len(self.items[tracker_ind])
                if len_tracker_item <= TRACKER_BUFFER_LENGTH:
                    temp_tracker[tracker_ind] = self.items[tracker_ind]
                else:
                    temp_tracker[tracker_ind] = self.items[tracker_ind][len_tracker_item - TRACKER_BUFFER_LENGTH:]

        return temp_tracker

    def format(self):
        self.items = {}
        self.total = 0

    def update(self, cnt, rect_list, d_list):
        # ---------------- check the pair of the closest rects  ---------------
        list_pair_update = self.__get_closest_pair(rect_list, cnt)
        total_in = 0
        total_out = 0
        # ---------------------- update/add new items -------------------------
        for i in range(len(rect_list)):
            if i in list_pair_update[0]:    # update
                tracker_ind = list_pair_update[1][list_pair_update[0].index(i)]
                cnt_in, cnt_out = self.__update_tracker(tracker_ind, cnt, rect_list[i], d_list[i])
                total_in += cnt_in
                total_out += cnt_out
            else:
                self.total += 1
                self.__add_tracker(cnt, rect_list[i], d_list[i])

        # ------------------------ delete the old items -----------------------
        self.items = self.__delete_tracker(cnt)

        return total_in, total_out
