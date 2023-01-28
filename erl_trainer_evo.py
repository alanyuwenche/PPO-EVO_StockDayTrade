
import numpy as np, os, time, random, torch, sys
from algos.neuroevolution import SSNE
from core import utils
from core.runner import rollout_worker
from torch.multiprocessing import Process, Pipe, Manager
import torch


class ERL_Trainer:

	def __init__(self, args, model_constructor, env_constructor):

		self.args = args
		self.policy_string = 'CategoricalPolicy' if env_constructor.is_discrete else 'Gaussian_FF'
		self.manager = Manager()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		#Evolution
		self.evolver = SSNE(self.args)

		#Initialize population
		self.population = self.manager.list()
		for _ in range(args.pop_size):
			self.population.append(model_constructor.make_model(self.policy_string))

		#Save best policy
		self.best_policy = model_constructor.make_model(self.policy_string)
		#state_dict = torch.load('/content/Evolutionary-Reinforcement-Learning/ppoS-113400.pth')
		#self.best_policy.load_state_dict(state_dict)#20221228 載入預先訓練的模型


		############## MULTIPROCESSING TOOLS ###################
		#Evolutionary population Rollout workers
		self.evo_task_pipes = [Pipe() for _ in range(args.pop_size)]
		self.evo_result_pipes = [Pipe() for _ in range(args.pop_size)]
		self.evo_workers = [Process(target=rollout_worker, args=(id, 'evo', self.evo_task_pipes[id][1], self.evo_result_pipes[id][0], args.rollout_size > 0, self.population, env_constructor)) for id in range(args.pop_size)]
		for worker in self.evo_workers: worker.start()
		self.evo_flag = [True for _ in range(args.pop_size)]

		#Test bucket
		self.test_bucket = self.manager.list()
		self.test_bucket.append(model_constructor.make_model(self.policy_string))

		# Test workers
		self.test_task_pipes = [Pipe() for _ in range(args.num_test)]
		self.test_result_pipes = [Pipe() for _ in range(args.num_test)]
		#self.test_workers = [Process(target=rollout_worker, args=(id, 'test', self.test_task_pipes[id][1], self.test_result_pipes[id][0], False, self.test_bucket, env_constructor)) for id in range(args.num_test)]
		#20220520 底下一行配合測試時選模型
		self.test_workers = [Process(target=rollout_worker, args=(id, 'test', self.test_task_pipes[id][1], self.test_result_pipes[id][0], True, self.test_bucket, env_constructor)) for id in range(args.num_test)]
		for worker in self.test_workers: worker.start()
		self.test_flag = False

		#Trackers
		self.best_score = -float('inf'); self.gen_frames = 0; self.total_frames = 0; self.test_score = None; self.test_std = None


	def forward_generation(self, gen, tracker):

		gen_max = -float('inf')

		#Start Evolution rollouts
		if self.args.pop_size > 1:
			for id, actor in enumerate(self.population):
				self.evo_task_pipes[id][0].send(id)

		#Start Test rollouts
		if gen % self.args.test_frequency == 0:
			self.test_flag = True
			for pipe in self.test_task_pipes: pipe[0].send(0) #20200520 pipe的物件結構為tuple- (connection, connection)




			#self.gen_frames = 0


		########## JOIN ROLLOUTS FOR EVO POPULATION ############
		all_fitness = []; all_eplens = []
		if self.args.pop_size > 1:
			for i in range(self.args.pop_size):
				s, fitness, frames, trajectory = self.evo_result_pipes[i][1].recv()
				#if len(trajectory) > 200: fitness -= 10*len(trajectory)#20220530 縮短交易長度
				all_fitness.append(fitness); all_eplens.append(frames)
				self.gen_frames+= frames; self.total_frames += frames
				self.best_score = max(self.best_score, fitness)
				gen_max = max(gen_max, fitness)

		######################### END OF PARALLEL ROLLOUTS ################

		############ FIGURE OUT THE CHAMP POLICY AND SYNC IT TO TEST #############
		if self.args.pop_size > 1:
			champ_index = all_fitness.index(max(all_fitness))
			utils.hard_update(self.test_bucket[0], self.population[champ_index])#self.population[champ_index]為網路結構(f1,f2,val,adv)
			if max(all_fitness) > self.best_score:
				self.best_score = max(all_fitness)
				utils.hard_update(self.best_policy, self.population[champ_index])
				torch.save(self.population[champ_index].state_dict(), self.args.aux_folder + '_best'+self.args.savetag)
				print("Best policy saved with score", '%.2f'%max(all_fitness))

		else: #If there is no population, champion is just the actor from policy gradient learner
			utils.hard_update(self.test_bucket[0], self.rollout_bucket[0])

		###### TEST SCORE ######
		if self.test_flag:
			self.test_flag = False
			test_scores = []
			test_N = 0  #有交易且長度小於200 20220520
			no_T = 0  #無交易次數 20220527
			#infos = [] #20220523
			for pipe in self.test_result_pipes: #Collect all results
				#_, fitness, _, _ = pipe[1].recv()
				_, fitness, fr, traj = pipe[1].recv() #20220520 配合測試時選模型-若當天沒任何動作: fitness=0,traj=280
				#infos.append(traj[-1][5])#20220523
				self.best_score = max(self.best_score, fitness)
				gen_max = max(gen_max, fitness)
				test_scores.append(fitness)
				if (abs(fitness) > 5) and (len(traj) < 200): test_N += 1  #20220520
				if (abs(fitness) < 5) and (len(traj) > 260): no_T += 1  #20220618, 20220527
			test_scores = np.array(test_scores)
			test_mean = np.mean(test_scores); test_std = (np.std(test_scores))
			tracker.update([test_mean], self.total_frames)
			
			if (test_N > 4) and (no_T > 1):
			#if test_N > 6:
				f = open("./data/logfile.txt","a")
				f.write('Gen: %d\t' % gen)
				f.write("test_N: %d\t" % test_N)
				f.write("test_mean: %d\t" % test_mean)
				f.write("test_std: %d\t" % test_std)
				f.write("no_T: %d\t" % no_T)
				#f.write("\n")#20220523
				#f.write("infos: %s\t" % np.array(infos))#20220523
				f.write("\n")
				f.close()
				fileN = './data/Gen-'+str(gen)+'.pth'
				torch.save(self.test_bucket[0].state_dict(),fileN)
			
		else:
			test_mean, test_std = None, None


		#NeuroEvolution's probabilistic selection and recombination step
		if self.args.pop_size > 1:
			self.evolver.epoch(gen, self.population, all_fitness)

		#Compute the champion's eplen #champ_len: 在MC開始為200
		champ_len = all_eplens[all_fitness.index(max(all_fitness))]
		#在這輸出的gen_max是經過L118,L128,L154比較後的最大值
		return gen_max, champ_len, all_eplens, test_mean, test_std


	def train(self, frame_limit):
		os.makedirs('./data/', exist_ok=True)
		# Define Tracker class to track scores
		test_tracker = utils.Tracker(self.args.savefolder, ['score_' + self.args.savetag], '.csv')  # Tracker class to log progress
		time_start = time.time()

		for gen in range(1, 1000000000):  # Infinite generations

			# Train one iteration
			max_fitness, champ_len, all_eplens, test_mean, test_std = self.forward_generation(gen, test_tracker)
			if test_mean: self.args.writer.add_scalar('test_score', test_mean, gen)

			print('Gen/Frames:', gen,'/',self.total_frames,
				  ' Gen_max_score:', '%.2f'%max_fitness,
				  ' Champ_len', '%.2f'%champ_len, ' Test_score u/std', utils.pprint(test_mean), utils.pprint(test_std))

			if gen % 5 == 0:
				print('Best_score_ever:''/','%.2f'%self.best_score, ' FPS:','%.2f'%(self.total_frames/(time.time()-time_start)), 'savetag', self.args.savetag)
				print(' Time:','%.2f'%((time.time()-time_start)),' sec')
				print()

			if self.total_frames > frame_limit:
				break

		###Kill all processes
		try:
			for p in self.task_pipes: p[0].send('TERMINATE')
			for p in self.test_task_pipes: p[0].send('TERMINATE')
			for p in self.evo_task_pipes: p[0].send('TERMINATE')
		except:
			None