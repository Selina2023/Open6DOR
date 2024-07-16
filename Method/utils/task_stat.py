import glob

paths = glob.glob('Method/output/rot_banch_0704/*/*/*/task_config.json')

print("total tasks", len(paths))
position_tags = set([paths[i].split('/')[-4] for i in range(len(paths))])
print(position_tags)
for position_tag in position_tags:
    print(position_tag, len([paths[i] for i in range(len(paths)) if paths[i].split('/')[-4] == position_tag]))