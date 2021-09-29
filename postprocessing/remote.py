import gzip

from fabric.connection import Connection


#  with Connection("copy") as c, c.sftp() as sftp,   \
#           sftp.open('.ssh/id_rsa') as file:
#      for line in file:
#          print(line)

#  with Connection("copy") as c, c.sftp() as sftp,   \
#           sftp.open('/rwthfs/rz/cluster/work/rwth0773/Islam96/polycrystal/mmc/test_cation/LSF_supercell_py.lmp.gz') as file:
#      file.prefetch()
#      file = file.read()
#      file = gzip.open(file)
#      print(type(file))

with Connection("copy") as c, c.sftp() as sftp:
    sftp.chdir('/home/rwth0773/03_MD/Islam96-rigid/LSF00/ortho/')

    import numpy as np
    import pandas as pd
    from scipy import constants as sc
    import seaborn as sns

    # from chemview.utils import get_atom_color
    from matplotlib import pyplot as plt
    from pymatgen.io.lammps.outputs import parse_lammps_log

    # def atom_col(atom):
    #     return '#{}'.format(str(format(get_atom_color(atom), 'x')))

    temps = [1000, 1200, 1400, 1600, 1800, 2000]  # 2000, 2600
    # start_fit = 1000
    # coefs_l = []
    # stats_l = []

    fig, ax = plt.subplots()
    ax.set_xlabel(r'$time$ / ps')
    ax.set_ylabel(r'$E \rm eV$')
    palette = sns.color_palette("magma", len(temps))

    for i, temp in enumerate(temps):
        with sftp.open("{}K/log.lammps.gz".format(temp)) as file:
            file.prefetch()
            df = parse_lammps_log(filename=gzip.open(file, mode='rt'))
            df = df[0]

        # with io.BytesIO() as fl:
        #     sftp.getfo("{}K/log.lammps.gz".format(temp), fl)
        #     fl.seek(0)
        #     filename = fl.getvalue()
        #     print(filename)
        #     df = parse_lammps_log(filename=filename)
        #     df = df[0]


        x = df['Time']
        z = df['TotEng'].ewm(span=10).mean()

        # coefs, stats = np.polynomial.polynomial.polyfit(x[start_fit:], df['c_msd4[4]'].iloc[start_fit:], 1, full=True)
        # coefs_l.append(coefs)
        # stats_l.append(stats)

        ax.plot(x, z, linewidth=1, label='{} K'.format(temp), color=palette[i])
        # ax.plot([x.iloc[start_fit], x.iloc[-1]], np.polynomial.polynomial.polyval([x.iloc[start_fit], x.iloc[-1]], coefs),
        #         color='k', linewidth=1)

    # ax.legend(loc='upper left')
    # fig.savefig("ploteng.pdf", bbox_inches='tight')
    plt.show()
