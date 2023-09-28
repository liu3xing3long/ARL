import json
import os
import pyarrow as pa


def build_mimic_from_KADCaption():
    input_json = './KADcaption.json'
    splits = {'train': 0.8, 'val': 0.1, 'test': 0.1}

    with open(input_json, 'r') as fp:
        d = json.load(fp)
        len_d = len(d)

        cur_it = 0
        for sp, spp in splits.items():
            this_it = int(spp * len_d)

            print(fr'data {sp}, length {this_it}')    
            spdata = d[cur_it: cur_it + this_it]
            cur_it += this_it

            output_json = []
            for spd in spdata:
                output_json.append({
                    'caption': ' '.join(spd['caption']),
                    'image_id': os.path.basename(spd['image']).strip('.jpg'), 
                    'image': os.path.basename(spd['image']),
                })

            with open(fr'./BLIP2anno/mimic_{sp}.json', 'w') as fp:
                json.dump(output_json, fp, indent=4)
    

def build_from_arrow(dataset, splits=None,input_root_path='../data/pretrain_arrows_umls/', output_path='../data/BLIP2anno/'):
    if splits is None:
        splits = ['train', 'val', 'test']

    if not os.path.exists(output_path):
            os.makedirs(output_path)

    for sp in splits:
        dataname = fr'{dataset}_{sp}'
        table = pa.ipc.RecordBatchFileReader(pa.memory_map(f"{input_root_path}/{dataname}.arrow", "r")).read_all()
        print(fr'dataset {dataset}, split {sp}, length {len(table)}')

        with open(fr'{output_path}/{dataname}.json', 'w') as fp:
            json.dump(
                [
                    {
                        'caption': ' '.join(map(str, cap)),
                        'image_id': os.path.splitext(os.path.basename(str(ipath)))[0], 
                        'image': os.path.basename(str(ipath)),
                    } for cap, ipath in zip(table['caption'], table['image_id'])
                ], fp, indent=4)
        

if __name__ == '__main__':
    build_from_arrow('mimic_cxr')
    build_from_arrow('roco')
    build_from_arrow('medicat')


