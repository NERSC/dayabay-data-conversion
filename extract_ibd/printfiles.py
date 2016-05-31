import os
base = '/global/homes/y/ynakajim/mywork'
base += '/data_ana_p14a/outputs_ibd_3sites_dbd'

ndays = 705
for EH in ['EH1', 'EH2', 'EH3']:
    path = os.path.join(base, EH, 'output_ibd_')
    for i in range(ndays):
        number = '%03d' % i
        output = path + number + '.root'
        print output
