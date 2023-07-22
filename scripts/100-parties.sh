for partition in noniid-labeldir
do
	for alg in fedprox/scaffold/fedcurv/fedmas
	do
		python run.py --model=simple-cnn \
			--dataset=mnist \
			--alg=$alg \
			--lr=0.01 \
			--batch-size=10 \
			--epochs=5 \
			--n_parties=100 \
			--rho=0.9 \
			--comm_round=50 \
			--partition=$partition \
			--beta=0.3\
			--device='cuda:0'\
			--datadir='./data/' \
			--logdir='./logs/' \
			--sample=0.1\
			--init_seed=0
	done

	for alg in fedprox/scaffold/fedcurv/fedmas
	do
		python experiments.py --model=cifar-net \
			--dataset=cifar10 \
			--alg=$alg \
			--lr=0.01 \
			--batch-size=10 \
			--epochs=5 \
			--n_parties=100 \
			--rho=0.9 \
			--comm_round=50 \
			--partition=$partition \
			--beta=0.3\
			--device='cuda:0'\
			--datadir='./data/' \
			--logdir='./logs/' \
			--sample=0.1\
			--init_seed=0
	done
done
