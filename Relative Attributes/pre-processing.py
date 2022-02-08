import os

root_path = '/Users/kun/Desktop/workspace/'  #  
data_path = root_path +'esd_en/'  # 待处理的音频路径

count = 0

opensmile_path = '/Users/kun/Desktop/workspace/open_smile/opensmile/build/progsrc/smilextract/SMILExtract'
config_path = '/Users/kun/Desktop/workspace/open_smile/opensmile/config/is09-13/IS09_emotion.conf'

error_list = []

for root, spk_dir, files in os.walk(data_path):
    # print('----')
    # print(root)
    print(spk_dir)
    # print(files)
    # exit()
    for spk in spk_dir:
        spk_out_dir  = root_path + 'output/' + spk
        if not os.path.exists(spk_out_dir):
            os.mkdir(spk_out_dir)
        for root, emo_dir, files in os.walk(data_path + spk):
            print(emo_dir)
            for emo in emo_dir:
                for _, _, wavfiles in os.walk(data_path + spk + '/' + emo):
                    for i in range(len(wavfiles)):
                        #print(count)
                        wavfile_path = data_path + spk + '/' + emo + '/' + wavfiles[i]
                        print(wavfile_path)
                        
                        emo_out_dir  = root_path + 'output/' + spk+ '/' + emo
                        print(emo_out_dir)
                        feature_path = emo_out_dir + '/' + wavfiles[i][:-4] + '.csv'
                        print(feature_path)
                        # path_remake(files[i])
                        try:
                            if not os.path.exists(emo_out_dir):
                                os.mkdir(emo_out_dir)
                            os.system(opensmile_path + ' -C ' + config_path + ' -I ' + wavfile_path + ' -csvoutput ' + feature_path + ' -instname ' + feature_path[-15:-4])
                        except:
                            error_list.append(wavfile_path)
                            count += 1
                        # exit()
print('error num: ',count)  # 出错次数，正常情况都是不会出错的。
print('error list: ', error_list)