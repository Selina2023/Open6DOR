
import json, glob

def get_assets_info(dataset_names):
    urdf_paths = []
    obj_name = []
    uuids = []
    if "ycb" in dataset_names:
        # all the ycb urdf data
        json_dict = json.load(open("../Benchmark/benchmark_catalogue/object_dictionary_complete_0702.json"))
        all_uuid = json_dict.keys()
        
        #ycb_urdf_paths = glob.glob("assets/ycb_16k_backup/*/*.urdf")
        ycb_urdf_paths = glob.glob("benchmark/mesh/ycb/*/*.urdf")
        ycb_names = [urdf_path.split("/")[-2] for urdf_path in ycb_urdf_paths]
        ycb_obj_name = [" ".join(name.split("_")[1:-2]) for name in ycb_names]
        ycb_uuid = [urdf_path.split("/")[-2].split("_")[0] for urdf_path in ycb_urdf_paths]
        
        valid_idx = [i for i in range(len(ycb_uuid)) if ycb_uuid[i] in all_uuid]
        
        ycb_uuids = [ycb_uuid[i] for i in valid_idx]
        ycb_urdf_paths = [ycb_urdf_paths[i] for i in valid_idx]
        ycb_obj_name = [" ".join(json_dict[ycb_uuid[i]]['category'].split("_")) for i in valid_idx]
        urdf_paths+=ycb_urdf_paths
        obj_name+=ycb_obj_name
        uuids += ycb_uuids
    if "objaverse" in dataset_names:
        json_dict = json.load(open("../Benchmark/benchmark_catalogue/object_dictionary_complete_0702.json"))
        
        all_uuid = json_dict.keys()
        # all the objaverse data
        objaverse_urdf_paths = glob.glob("assets/objaverse_final_norm/*/*_2.urdf")
        objaverse_obj_uuid = [path.split("/")[-2] for path in objaverse_urdf_paths]
        
        valid_idx = [i for i in range(len(objaverse_obj_uuid)) if objaverse_obj_uuid[i] in all_uuid]
        objaverse_obj_uuids = [objaverse_obj_uuid[i] for i in valid_idx]
        objaverse_urdf_paths = [objaverse_urdf_paths[i] for i in valid_idx]
        objaverse_obj_name = [" ".join(json_dict[objaverse_obj_uuid[i]]['category'].split("_")) for i in valid_idx]
        urdf_paths+=objaverse_urdf_paths
        obj_name+=objaverse_obj_name
        uuids+=objaverse_obj_uuids
    if "objaverse_old" in dataset_names:
        json_dict = json.load(open("category_dictionary.json"))
        
        all_uuid = []
        for key in json_dict.keys(): all_uuid+=json_dict[key]["object_uuids"]
        # all the objaverse data
        objaverse_urdf_paths = glob.glob("benchmark/mesh/objaverse_final_norm/*/*_2.urdf")
        objaverse_names = [urdf_path.split("/")[-2] for urdf_path in objaverse_urdf_paths]
        objaverse_obj_name = [" ".join(name.split("_")[1:]) for name in objaverse_names]
        objaverse_obj_uuid = [name.split("_")[0] for name in objaverse_names]
        valid_idx = [i for i in range(len(objaverse_obj_uuid)) if objaverse_obj_uuid[i] in all_uuid]
        objaverse_urdf_paths = [objaverse_urdf_paths[i] for i in valid_idx]
        objaverse_obj_name = [objaverse_obj_name[i] for i in valid_idx]
        # import pdb; pdb.set_trace()
        urdf_paths+=objaverse_urdf_paths
        obj_name+=objaverse_obj_name
    return urdf_paths,obj_name,uuids