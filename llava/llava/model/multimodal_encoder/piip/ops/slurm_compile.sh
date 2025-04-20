srun -p Intern5 --quotatype spot -n 1 --gres gpu:1 --ntasks-per-node 1 python setup.py build install

