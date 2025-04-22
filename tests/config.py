ANIMAL_PARAM = {        
    'target_pix_per_mm': 5,
    'intensity': 0.15,
    'gamma': 1.0,
    'contrast': 1.0,
    'min_size_mm': 0.0,
    'max_size_mm': 300.0,
    'min_length_mm': 0,
    'max_length_mm': 0,
    'min_width_mm': 0,
    'max_width_mm': 0,
    'blur_sz_mm': 0.6,
    'median_filter_sz_mm': 0,
    'downsample_factor': 0.90,
    'crop_dimension_mm': (0,0), 
    'crop_offset_y_mm': 0
}

BODY_PARAM = {
    'target_pix_per_mm': 10,
    'intensity': 0.15,
    'gamma': 1.0,
    'contrast': 3.0,
    'min_size_mm': 2.0,
    'max_size_mm': 300.0,
    'min_length_mm': 0,
    'max_length_mm': 0,
    'min_width_mm': 0,
    'max_width_mm': 0,
    'blur_sz_mm': 0.6,
    'median_filter_sz_mm': 0,
    'crop_dimension_mm': (5,5),
    'crop_offset_y_mm': 0
}

EYES_PARAM = {
    'target_pix_per_mm': 40,
    'thresh_lo': 0.2,
    'thresh_hi': 1.0,
    'gamma': 2.0,
    'dyntresh_res': 10,
    'contrast': 5.0,
    'size_lo_mm': 0.1,
    'size_hi_mm': 30.0,
    'blur_sz_mm': 0.1,
    'median_filter_sz_mm': 0,
    'crop_dimension_mm': (1,1.5),
    'crop_offset_y_mm': -0.25
}

TAIL_PARAM = {
    'target_pix_per_mm': 20,
    'ball_radius_mm': 0.1,
    'arc_angle_deg': 90,
    'n_tail_points': 6,
    'n_pts_arc': 20,
    'n_pts_interp': 40,
    'tail_length_mm': 3.0,
    'blur_sz_mm': 0.06,
    'median_filter_sz_mm': 0,
    'contrast': 3.0,
    'gamma': 0.75,
    'crop_dimension_mm': (3.5,3.5),
    'crop_offset_y_mm': 2
}