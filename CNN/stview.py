import pickle
import math
import numpy
import requests
import io
import time

import PIL
import PIL.Image

import cv2
import skimage
import skimage.exposure
import skimage.transform

import colorsys
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt

import theano
import theano.tensor
import theano.tensor as T

import googlemaps
import googlemaps.client
import googlemaps.convert
import googlemaps.directions
import googlemaps.distance_matrix
import googlemaps.elevation
import googlemaps.exceptions
import googlemaps.geocoding
import googlemaps.roads
import googlemaps.timezone

import polyline
import polyline.codec

import webbrowser
import pygmaps.pygmaps

import CNN
import CNN.prop
import CNN.conv
import CNN.utils
import CNN.enums
import CNN.nms


class StreetViewSpan:
    def __init__(self, load_models=True):

        self.api_key = self.__read_api_key()
        self.__load_models = load_models
        if not load_models:
            return

        print("... start building the models")
        t1 = time.clock()

        self.__batch_size = 50
        self.__img_dim_80 = 80
        self.__img_dim_28 = 28

        # some images needed for visualization
        img_sc_prohib = cv2.imread("D:\\_Dataset\\UK\\sc_prohibitory.png", cv2.IMREAD_UNCHANGED)
        img_sc_mandat = cv2.imread("D:\\_Dataset\\UK\\sc_mandatory.png", cv2.IMREAD_UNCHANGED)
        img_sc_warning = cv2.imread("D:\\_Dataset\\UK\\sc_warning.png", cv2.IMREAD_UNCHANGED)
        self.__sc_imgs = (img_sc_warning, img_sc_prohib, img_sc_mandat)

        # load the models once and for all
        prohib_recog_model_path = "D:\\_Dataset\\GTSRB\\cnn_model_p_80.pkl"
        mandat_recog_model_path = "D:\\_Dataset\\GTSRB\\cnn_model_m_80.pkl"
        prohib_detec_model_path = "D:\\_Dataset\\GTSDB\\las_model_p_80_binary.pkl"
        mandat_detec_model_path = "D:\\_Dataset\\GTSDB\\las_model_m_80_binary.pkl"
        superclass_recognition_model_path = "D:\\_Dataset\\SuperClass\\cnn_model_28_lasagne.pkl"

        # build the model once and for all
        self.__detect_net_p = self.__build_detector(prohib_recog_model_path, prohib_detec_model_path, self.__batch_size)
        self.__detect_net_m = self.__build_detector(mandat_recog_model_path, mandat_detec_model_path, self.__batch_size)
        self.__recog_superclass_cnn = self.__build_classifier(superclass_recognition_model_path)

        t2 = time.clock()
        duration = t2 - t1
        print("... finish building the models, time(sec.): %f" % (duration))

    def span_google_street_view(self, address_from="", address_to=""):

        client = googlemaps.client.Client(key=self.api_key)

        # convert start/stop addresses to geo-locations
        address11 = '6 Longmead Road, Townhill Park, Southampton, UK'
        address12 = 'Portswood Street, Southampton, UK'
        address13 = 'Sainsbury\'s, 224 Portswood Road, Southampton SO17 2LB, United Kingdom'

        address21 = 'University of Southampton, Highfield Campus, Southampton, UK'
        address22 = 'Jurys Inn Southampton, Charlotte Place, Southampton SO14 0TB, United Kingdom'

        if len(address_from) == 0 or len(address_to) == 0:
            address_from = address22
            address_to = address13

        geocode_start = googlemaps.client.geocode(client, address_from)
        geocode_stop = googlemaps.client.geocode(client, address_to)

        start_location = geocode_start[0]["geometry"]["location"]
        stop_location = geocode_stop[0]["geometry"]["location"]

        # get the direction from start to stop, get them in terms of geo-location points // driving
        direction_result = googlemaps.client.directions(client, start_location, stop_location, mode="driving")

        # decode the polyline of the direction to get the points
        points = polyline.codec.PolylineCodec().decode(direction_result[0]["overview_polyline"]["points"])
        locations = self.__convert_points_to_locations(points)
        locations = self.__calculate_heading(locations)
        n_locations = len(locations)
        locations = locations[- (n_locations - 10):]

        # missing steps
        # generate more points, adjust the pace, then calculate the heading
        meters_per_frame = 5
        locations = self.__augument_path(locations, meters_per_frame)
        self.__plot_points_on_map(locations)
        self.__show_street_view_images(locations, self.api_key)

        #road_locations = googlemaps.client.snap_to_roads(client,, interpolate = True)

        # loc1 = locations[3]
        # loc2 = locations[4]
        # distance = self.__measure_distance(loc1, loc2)
        # frames = int(distance / meters_per_frame)
        # print("frames: %d" % frames)
        # new_locations = self.__interpolate_path(loc1, loc2, frames)
        # print("start: %f, %f" % (loc1["lat"], loc1["lng"]))
        # print("stop: %f, %f" % (loc2["lat"], loc2["lng"]))
        # for loc in new_locations:
        #     print("%f, %f" % (loc["lat"], loc["lng"]))
        # self.__plot_points_on_map(new_locations)

        # self.__show_street_view_images(locations, self.api_key)

        # since we interpolated points in the direction, these generated points might not be on
        # the road (if road wasn't straight line). The solution is to snap these point to the road
        # path = [(start_location_lat, start_location_lng), (stop_location_lat, stop_location_lng)]
        # road_locations = googlemaps.client.snap_to_roads(client, path, interpolate=True)

        dummy_object = True

    def process_image_and_save(self, img_path, count):
        if not self.__load_models:
            print("Sorry, can't process image because models were not loaded!!!!")
            return

        img_color = cv2.imread(img_path)
        img_result = self.__process_image(img_color)
        if img_result is not None:
            img_path = "D://_Dataset//GTSDB//Test_Regions/result_%d.png" % (count)
            cv2.imwrite(img_path, img_result)

    def __process_image(self, img_color):

        t1 = time.clock()

        # detect the region, using superclass-specific recognition model
        detec_result_p = self.__detect(img_color, self.__batch_size, self.__detect_net_p)
        detec_result_m = self.__detect(img_color, self.__batch_size, self.__detect_net_m)

        regions = []
        if len(detec_result_p[0]) > 0:
            for r in detec_result_p[0]:
                regions.append(r)
        if len(detec_result_m[0]) > 0:
            for r in detec_result_m[0]:
                regions.append(r)
        if len(regions) == 0:
            print("... NO TRAFFIC SIGN FOUND BY THE DETECTORS")
            return None

        # merge only the strong regions from the detector
        # then create different superclass regions (at different scales)
        # to be passed to the super_class classifier
        regions = numpy.vstack(regions)
        weak_region, regions = CNN.nms.suppression(regions, 0.25, 0)
        scales = numpy.arange(0.9, 1.3, 0.1)
        n_scales = len(scales)
        sc_regions = self.__regions_at_different_scales(img_color, self.__img_dim_28, regions, scales)
        print("... start classify superclass")
        sc_prediction = self.__classify_images(self.__recog_superclass_cnn, sc_regions, self.__img_dim_28)
        print("... finish classify superclass")

        # we've originally let's say 4 strong regions detected by the detector
        # then we created sc_regions for these 4 regions, with different 3 scales
        # so the result is 4 * 3 = 12 regions. Now for each of the prediction results
        # deal with each of 3 of them alone, the mean of their prediction
        # is the final prediction of this region
        superclass_ids = []
        for i in range(0, len(regions)):
            predictions = sc_prediction[i * n_scales: (i + 1) * n_scales]
            occurrence = []
            for i in [0, 1, 2]:
                occurrence.append(predictions.tolist().count(i))
            if occurrence[0] >= occurrence[1] and occurrence[0] >= occurrence[2]:
                superclass_id = 0
            elif occurrence[1] >= occurrence[0] and occurrence[1] >= occurrence[2]:
                superclass_id = 1
            else:
                superclass_id = 2
            superclass_ids.append(superclass_id)

        t2 = time.clock()
        duration = t2 - t1
        print("... finish processing the image, time(sec.): %d" % (duration))

        # now, we have the regions and the prediction (class id) of the superclasses in the image
        img_result = self.__draw_superclass_result(img_color, regions, superclass_ids, self.__sc_imgs)
        return img_result

    # region Detector

    def __build_detector(self, recognition_model_path, detection_model_path, batch_size):
        # stack the regions of all the scales in one array
        # please note that a scale can have no regions, so using vstack wouldn't work
        # remove the scales with empty regions then use vstack

        ##############################
        # Build the detector         #
        ##############################

        loaded_objects = CNN.utils.load_model(model_path=recognition_model_path, model_type=CNN.enums.ModelType._02_conv3_mlp2)
        img_dim = loaded_objects[1]
        kernel_dim = loaded_objects[2]
        nkerns = loaded_objects[3]
        pool_size = loaded_objects[5]

        layer0_W = theano.shared(loaded_objects[6], borrow=True)
        layer0_b = theano.shared(loaded_objects[7], borrow=True)
        layer1_W = theano.shared(loaded_objects[8], borrow=True)
        layer1_b = theano.shared(loaded_objects[9], borrow=True)
        layer2_W = theano.shared(loaded_objects[10], borrow=True)
        layer2_b = theano.shared(loaded_objects[11], borrow=True)

        layer0_input = T.tensor4(name='input')
        layer0_img_dim = img_dim
        layer0_img_shape = (batch_size, 1, layer0_img_dim, layer0_img_dim)
        layer0_kernel_dim = kernel_dim[0]
        layer1_img_dim = int((layer0_img_dim - layer0_kernel_dim + 1) / 2)
        layer1_kernel_dim = kernel_dim[1]
        layer2_img_dim = int((layer1_img_dim - layer1_kernel_dim + 1) / 2)
        layer2_kernel_dim = kernel_dim[2]
        layer3_img_dim = int((layer2_img_dim - layer2_kernel_dim + 1) / 2)
        layer3_input_shape = (batch_size, nkerns[2] * layer3_img_dim * layer3_img_dim)

        # layer 0, 1, 2: Conv-Pool
        layer0_output = CNN.conv.convpool_layer(
            input=layer0_input, W=layer0_W, b=layer0_b,
            image_shape=(batch_size, 1, layer0_img_dim, layer0_img_dim),
            filter_shape=(nkerns[0], 1, layer0_kernel_dim, layer0_kernel_dim),
            pool_size=pool_size
        )
        layer1_output = CNN.conv.convpool_layer(
            input=layer0_output, W=layer1_W, b=layer1_b,
            image_shape=(batch_size, nkerns[0], layer1_img_dim, layer1_img_dim),
            filter_shape=(nkerns[1], nkerns[0], layer1_kernel_dim, layer1_kernel_dim),
            pool_size=pool_size
        )
        layer2_output = CNN.conv.convpool_layer(
            input=layer1_output, W=layer2_W, b=layer2_b,
            image_shape=(batch_size, nkerns[1], layer2_img_dim, layer2_img_dim),
            filter_shape=(nkerns[2], nkerns[1], layer2_kernel_dim, layer2_kernel_dim),
            pool_size=pool_size
        )
        # do the filtering using 3 layers of Conv+Pool
        conv_fn = theano.function([layer0_input], layer2_output)

        # load the regression model
        with open(detection_model_path, 'rb') as f:
            nn_mlp = pickle.load(f)

        return conv_fn, nn_mlp, layer3_input_shape

    def __detect(self, img_color, batch_size, net):
        """
        detect a traffic sign form the given natural image
        detected signs depend on the given model, for example if it is a prohibitory detection model
        we'll only detect prohibitory traffic signs
        :param img_path:
        :param model_path:
        :param classifier:
        :param img_dim:
        :return:
        """

        net_cnn, net_mlp, net_mlp_input_shape = net

        ##############################
        # Extract detection regions  #
        ##############################

        # pre-process image by: equalize histogram and stretch intensity
        # converting to gray-scale and normalizing are redundant steps
        img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        img = img.astype(float) / 255.0
        img_dim = 80

        # min, max defines what is the range the detection proposals
        max_window_dim = int(img_dim * 2)
        min_window_dim = int(img_dim / 4)

        # regions, locations and window_dim at each scale
        regions = []
        locations = []
        window_dims = []
        r_count = 0

        # important, instead of naively add every sliding window, we'll only add
        # windows that covers the strong detection proposals
        prop_weak, prop_strong, prop_map, prop_circles = CNN.prop.detection_proposal(img_color, min_dim=min_window_dim, max_dim=max_window_dim)
        if len(prop_strong) == 0:
            return [], [], [], []

        # loop on the detection proposals
        scales = numpy.arange(0.7, 1.58, 0.05)
        for prop in prop_strong:
            x1 = prop[0]
            y1 = prop[1]
            x2 = prop[2]
            y2 = prop[3]
            w = x2 - x1
            h = y2 - y1
            window_dim = max(h, w)
            center_x = int(x1 + round(w / 2))
            center_y = int(y1 + round(h / 2))

            for scale in scales:
                r_count += 1
                dim = window_dim * scale
                dim_half = round(dim / 2)
                dim = round(dim)
                x1 = center_x - dim_half
                y1 = center_y - dim_half
                x2 = center_x + dim_half
                y2 = center_y + dim_half

                # pre-process the region and scale it to img_dim
                region = img[y1:y2, x1:x2]
                region = skimage.transform.resize(region, output_shape=(img_dim, img_dim))
                region = skimage.exposure.equalize_hist(region)

                # we only need to store the region, it's top-left corner and sliding window dim
                regions.append(region)
                locations.append([x1, y1])
                window_dims.append(dim)

        ##############################
        # Start detection            #
        ##############################

        # split it to batches first, zero-pad them if needed
        n_regions = len(regions)
        if n_regions % batch_size != 0:
            n_remaining = batch_size - (n_regions % batch_size)
            regions_padding = numpy.zeros(shape=(n_remaining, img_dim, img_dim), dtype=float)
            regions = numpy.vstack((regions, regions_padding)).tolist()

        # run the detector on the regions
        start_time = time.clock()

        # loop on the batches of the regions
        n_regions_padded = len(regions)
        n_batches = int(n_regions_padded / batch_size)
        layer0_img_shape = (batch_size, 1, img_dim, img_dim)
        predictions = []
        for i in range(n_batches):
            # prediction: CNN filtering then MLP regression
            t1 = time.clock()
            batch = numpy.asarray(regions[i * batch_size: (i + 1) * batch_size])
            batch = batch.reshape(layer0_img_shape)
            filters = net_cnn(batch)
            filters = filters.reshape(net_mlp_input_shape).astype("float32")
            batch_pred = net_mlp.predict(filters)
            predictions.append(batch_pred)
            t2 = time.clock()
            print("... batch: %i/%i, time(sec.): %f" % ((i + 1), n_batches, t2 - t1))

        # after getting all the predictions, remove the padding
        predictions = numpy.hstack(predictions)
        predictions = predictions[0:n_regions]

        # now, here is the thing, since this function serves detection model
        # of prohibitory and mandatory, and both models were built slightly differently
        # so all we want to do is to convert the predictions into list of bool values
        if predictions.ndim != 1 and predictions.ndim != 2:
            raise Exception("There must be something wrong here, why the predictions of the detector has wrong dimension?")
        if predictions.ndim == 2:
            predictions = predictions.reshape((predictions.shape[0],))
            predictions[predictions >= 0.5] = 1
            predictions[predictions < 0.5] = 0
        predictions = predictions.astype(bool).tolist()

        end_time = time.clock()
        duration = (end_time - start_time)
        print("... detection regions: %d, duration(sec.): %f" % (r_count, duration))

        # construct the probability map for each scale and show it/ save it
        s_count = 0
        overlap_thresh = 0.5
        min_overlap = 0
        strong_prob_regions = []
        weak_prob_regions = []
        for pred, loc, window_dim in zip(predictions, locations, window_dims):
            s_count += 1
            w_regions, s_regions = self.__probability_map([pred], [loc], window_dim, overlap_thresh, min_overlap)
            if len(w_regions) > 0:
                weak_prob_regions.append(w_regions)
            if len(s_regions) > 0:
                strong_prob_regions.append(s_regions)

        if len(weak_prob_regions) > 0:
            weak_prob_regions = numpy.vstack(weak_prob_regions)

        if len(strong_prob_regions) > 0:
            strong_prob_regions = numpy.vstack(strong_prob_regions)

        # now, after we finished scanning at all the levels, we should make the final verdict
        # by suppressing all the strong_regions that we extracted on different scales
        if len(strong_prob_regions) > 0:
            overlap_thresh = 0.25
            min_overlap = round(len(scales) * 0.35)
            weak_regions, strong_regions = CNN.nms.suppression(strong_prob_regions, overlap_thresh, min_overlap)
            if weak_regions is not list:
                weak_regions.tolist()
            if strong_regions is not list:
                strong_regions.tolist()
            return strong_regions, weak_regions, strong_prob_regions, weak_prob_regions
        else:
            return [], [], [], []

    def __probability_map(self, predictions, locations, window_dim, overlap_thresh, min_overlap):
        locations = numpy.asarray(locations)
        predictions = numpy.asarray(predictions)

        regions = []
        idx = numpy.where(predictions)[0]
        for i in idx:
            region = [0, 0, window_dim, window_dim]
            location = locations[i]
            x1 = int(region[0] + location[0])
            y1 = int(region[1] + location[1])
            x2 = int(region[2] + location[0])
            y2 = int(region[3] + location[1])
            regions.append([x1, y1, x2, y2])

        # check if no region found
        if len(regions) == 0:
            return [], []

        # suppress the new regions and raw them with red color
        weak_regions, strong_regions = CNN.nms.suppression(regions, overlap_thresh, min_overlap)

        # return the map to be exploited later by the detector, for the next scale
        weak_regions = weak_regions.tolist()
        return weak_regions, strong_regions

    def __save_detection_result(self, img_color, regions, img_id):
        strong_regions = regions[0]
        weak_regions = regions[1]
        strong_probability_regions = regions[2]
        weak_probability_regions = regions[3]

        # draw the result of the detection
        red_color = (0, 0, 255)
        blue_color = (255, 0, 0)
        green_color = (0, 255, 0)
        yellow_color = (84, 212, 255)
        for reg in weak_probability_regions:
            cv2.rectangle(img_color, (reg[0], reg[1]), (reg[2], reg[3]), green_color, 1)
        for reg in strong_probability_regions:
            cv2.rectangle(img_color, (reg[0], reg[1]), (reg[2], reg[3]), blue_color, 1)
        for reg in weak_regions:
            cv2.rectangle(img_color, (reg[0], reg[1]), (reg[2], reg[3]), yellow_color, 1)
        for reg in strong_regions:
            cv2.rectangle(img_color, (reg[0], reg[1]), (reg[2], reg[3]), red_color, 2)

        img_path = "D://_Dataset//GTSDB//Test_Regions/result_detect_%d.png" % (img_id)
        cv2.imwrite(img_path, img_color)

    def __draw_superclass_result(self, img_color, regions, superclass_ids, imgs_sc):
        color_red = (0, 0, 255)
        color_blue = (255, 0, 0)
        color_yellow = (0, 255, 255)
        colors = [color_yellow, color_red, color_blue]

        img_width = img_color.shape[1]
        img_result = img_color.copy()

        for region, superclass_id in zip(regions, superclass_ids):
            x1, y1, x2, y2 = region
            color = colors[superclass_id]
            cv2.rectangle(img_result, (x1, y1), (x2, y2), color, 2)

            # also, we want to overlay the ground truth beside the region
            offset_big = 12
            offset_small = 6
            dim = max(y2 - y1, x2 - x1)
            ground_truth = imgs_sc[superclass_id]
            ground_truth = cv2.resize(src=ground_truth, dsize=(dim, dim), interpolation=cv2.INTER_AREA)
            if x1 - dim - offset_big > -1:
                img_result[y1:y2, x1 - dim - offset_small: x2 - dim - offset_small, :] = ground_truth
            elif x2 + dim + offset_big > img_width:
                img_result[y1:y2, x1 + dim + offset_small: x2 + dim + offset_small, :] = ground_truth

        return img_result

    def __regions_at_different_scales(self, img_color, img_dim, regions, scales):
        img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        new_regions = []

        for region in regions:
            x1 = region[0]
            y1 = region[1]
            x2 = region[2]
            y2 = region[3]
            w = x2 - x1
            h = y2 - y1
            window_dim = max(h, w)
            center_x = int(x1 + round(w / 2))
            center_y = int(y1 + round(h / 2))
            for scale in scales:
                dim_half = round(window_dim * scale / 2)
                x1 = center_x - dim_half
                y1 = center_y - dim_half
                x2 = center_x + dim_half
                y2 = center_y + dim_half

                # pre-process the region and scale it to img_dim
                region = img[y1:y2, x1:x2]
                region = skimage.transform.resize(region, output_shape=(img_dim, img_dim))
                region = skimage.exposure.equalize_hist(region)

                # we only need to store the region, it's top-left corner and sliding window dim
                new_regions.append(region)

        return new_regions

    # endregion

    # region Classifier

    def __build_classifier(self, model_path):
        # load the model
        with open(model_path, 'rb') as f:
            net_cnn = pickle.load(f)

        return net_cnn

    def __classify_img(self, net_cnn, img, img_dim):
        img = img.reshape((1, 1, img_dim, img_dim))
        prediction = net_cnn.predict(img)
        return prediction

    def __classify_images(self, net_cnn, images, img_dim):
        # cast as numpy if needed
        if not isinstance(images, numpy.ndarray):
            images = numpy.asarray(images)

        images = images.reshape((images.shape[0], 1, img_dim, img_dim))
        prediction = net_cnn.predict(images)
        return prediction

    # endregion

    # region View/Show/Plot

    def __show_street_view_images(self, directions, api_key):
        # loop on all the points and get the google street view image at each one
        plt.figure(num=1, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='w')
        plt.ion()
        plt.axis('off')
        plt.show()

        # download the images of google street view at each location/step
        img_count = 0
        for dir in directions:
            latlng = "%f,%f" % (dir["lat"], dir["lng"])
            heading = dir["heading"]
            url = "https://maps.googleapis.com/maps/api/streetview?size=640x400&location=%s&heading=%f&pitch=0&key=%s" % (latlng, heading, api_key)
            img_bytes = requests.get(url).content
            img = numpy.asarray(PIL.Image.open(io.BytesIO(img_bytes)))
            plt.imshow(img)
            plt.pause(0.1)
            img_count += 1
            print("... new image: %d" % (img_count))

    def __plot_points_on_map(self, locations, is_locations=True):
        if is_locations:
            points = self.__convert_locations_to_points(locations)
        else:
            points = locations
        zoom_level = 16
        p_count = len(points)
        center = points[int(p_count / 2)]
        mymap = pygmaps.pygmaps.maps(center[0], center[1], zoom_level)

        # mymap.setgrids(37.42, 37.43, 0.001, -122.15, -122.14, 0.001)
        # mymap.addradpoint(37.429, -122.145, 95, "#FF0000")

        # create range of colors for the points
        hex_colors = []
        for val in range(1, p_count + 1):
            col = self.__pseudo_color(val, 0, p_count)
            hex_colors.append(self.__rgb_to_hex(col))

        # draw marks at the points
        p_count = 0
        for pnt, col in zip(points, hex_colors):
            p_count += 1
            mymap.addpoint(pnt[0], pnt[1], col, title=str(p_count))

        # draw path using the points then show the map
        path_color = "#0A6491"
        mymap.addpath(points, path_color)
        mymap.draw('mymap.draw.html')
        url = 'mymap.draw.html'
        webbrowser.open_new_tab(url)

    def __draw_image(self, img, num):
        # plot original image and first and second components of output
        # plt.figure(num)
        # plt.gray()
        # plt.ion()
        # plt.axis('off')
        plt.imshow(img)
        plt.show()

    # endregion

    # region Path Calculation

    def __augument_path(self, locations, meters_per_frame=1):
        """
        Generate points between every 2 points in the given steps
        This is to enrich the points within the path
        :param direction_steps:
        :param frames_per_meter:
        :return:
        """

        new_locations = []
        for i in range(0, len(locations) - 1):
            loc1 = locations[i]
            loc2 = locations[i + 1]
            distance = self.__measure_distance(loc1, loc2)
            if distance < meters_per_frame:
                continue
            frames = int(distance / meters_per_frame)
            interpolated = self.__interpolate_path(loc1, loc2, frames)
            new_locations.append(interpolated)

        # update the directions so that when at a waypoint you're looking
        # towards the next
        if len(new_locations) > 0:
            new_locations = numpy.hstack(new_locations)
            new_locations = self.__calculate_heading(new_locations)

        return new_locations

    def __augument_path_old(self, direction_steps, frames_per_meter=1):
        """
        Generate points between every 2 points in the given steps
        This is to enrich the points within the path
        :param direction_steps:
        :param frames_per_meter:
        :return:
        """

        locations = []
        for i in range(0, len(direction_steps) - 1):
            step = direction_steps[i]
            distance = step["distance"]["value"]
            frames = int(distance * frames_per_meter)
            location1 = step["start_location"]
            location2 = step["end_location"]
            interpolated = self.__interpolate_path(location1, location2, frames)
            locations.append(interpolated)

        # update the directions so that when at a waypoint you're looking
        # towards the next
        locations = numpy.hstack(locations)
        locations = self.__calculate_heading(locations)

        return locations

    def __measure_distance(self, location1, location2):
        R = 6371  # earth's mean radius in km
        dLat = math.radians(location2["lat"] - location1["lat"])
        dLong = math.radians(location2["lng"] - location1["lng"])
        a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(math.radians(location1["lat"])) * math.cos(math.radians(location2["lat"])) * math.sin(dLong / 2) * math.sin(dLong / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        d = R * c * 1000
        # returned distance is in meters
        return d

    def __adjust_pace(self, locations):
        """
        For the given locations, check if the distance between each one is near to the give
        pace, if not either add or remove locations to adjust the pace
        After that, calculate the heading of each point such that each one
        is pointing at/looking towards it's next
        :param locations:
        :return:
        """

        adjusted_locations = []
        return adjusted_locations

    def __interpolate_path(self, location1, location2, frames):
        """
        Generate points between the points of the given start/stop points.
        :param lat1:
        :param lng1:
        :param lat2:
        :param lng2:
        :param frames:
        :return:
        """

        x = [location1["lat"], location2["lat"]]
        y = [location1["lng"], location2["lng"]]
        if x[0] - x[1] == 0:
            yvals = numpy.linspace(y[0], y[1], frames)
            xvals = numpy.interp(yvals, y, x)
        else:
            xvals = numpy.linspace(x[0], x[1], frames)
            yvals = numpy.interp(xvals, x, y)

        # create geo location point with each point
        # as dictionary containing lat and lng values
        locations = []
        for lat, lng in zip(xvals, yvals):
            point = {"lat": lat, "lng": lng}
            locations.append(point)

        return locations

    def __calculate_heading(self, locations):
        """
        For the given list of locations, calculate and add the heading for each of them
        :param locations:
        :return:
        """

        heading = 0
        n = len(locations)
        for i in range(0, n):
            loc = locations[i]
            if i < n - 1:
                heading = self.__compute_direction(loc, locations[i + 1])
            loc["heading"] = heading

        return locations

    def __compute_direction(self, point1, point2):
        lat1 = point1["lat"]
        lng1 = point1["lng"]
        lat2 = point2["lat"]
        lng2 = point2["lng"]
        lambda1 = math.radians(lng1)
        lambda2 = math.radians(lng2)
        psi1 = math.radians(lat1)
        psi2 = math.radians(lat2)

        y = math.sin(lambda2 - lambda1) * math.cos(psi2)
        x = math.cos(psi1) * math.sin(psi2) - math.sin(psi1) * math.cos(psi2) * math.cos(lambda2 - lambda1)
        return math.degrees(math.atan2(y, x))

    def __snap_result_to_locations(self, snap_result):
        locations = []
        for snap in snap_result:
            snap = snap["location"]
            loc = {"lat": snap["latitude"], "lng": snap["longitude"]}
            locations.append(loc)
        return locations

    # endregion

    # region Conversions

    def __convert_steps_to_locations(self, steps):
        locations = []
        for step in steps:
            loc = step["start_location"]
            locations.append(loc)
        return locations

    def __convert_locations_to_points(self, locations):
        points = []
        for loc in locations:
            points.append((loc["lat"], loc["lng"]))
        return points

    def __convert_points_to_locations(self, points):
        locations = []
        for p in points:
            location = {"lat": p[0], "lng": p[1]}
            locations.append(location)
        return locations

    # endregions

    # region Color Manipluation

    def __pseudo_color(self, val, minval, maxval):
        # convert val in range minval..maxval to the range 0..120 degrees which
        # correspond to the colors red..green in the HSV colorspace
        h = (float(val - minval) / (maxval - minval)) * 120
        # convert hsv color (h,1,1) to its rgb equivalent
        # note: the hsv_to_rgb() function expects h to be in the range 0..1 not 0..360
        r, g, b = colorsys.hsv_to_rgb(h / 360, 1., 1.)
        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)
        return r, g, b

    def __hex_to_rgb(self, value):
        value = value.lstrip('#')
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    def __rgb_to_hex(self, rgb):
        return '#%02x%02x%02x' % rgb

    # endregion

    # region Mics

    def __read_api_key(self):
        file_path = "C://Users//Noureldien//Documents//PycharmProjects//TrafficSignRecognition//Data//google-maps-key.pkl"
        key = pickle.load(open(file_path, "rb"))
        return key

    def __tutorial(self):
        api_key = self.__read_api_key()
        client = googlemaps.client.Client(key=api_key)

        # Geocoding and address
        geocode_result = googlemaps.client.geocode(client, '1600 Amphitheatre Parkway, Mountain View, CA')

        # Look up an address with reverse geocoding
        reverse_geocode_result = googlemaps.client.reverse_geocode(client, (40.714224, -73.961452))

        # Request directions via public transit
        now = googlemaps.client.datetime.now()
        directions_result = googlemaps.client.directions(client, "Sydney Town Hall", "Parramatta, NSW", mode="transit", departure_time=now)

    # endregion

    # this dummy variable to enable the last #region to collapse/expand
    __dummuy_var = 10
