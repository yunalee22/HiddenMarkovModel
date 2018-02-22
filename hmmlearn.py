import sys
import numpy as np
import json

""" Class to represent a Hidden Markov Model. """
class HMM():
	
	def __init__(self, lines):
		self.lines = lines
		self.smoothing_value = 0.5
		self.wordset = []
		self.tagset = []
		self.transition_probabilities = {}
		self.emission_probabilities = {}


	# Given training data, counts number of distinct words and tags
	def parseTrainingData(self):
		for line in self.lines:
			for entry in line.split():
				split = entry.rsplit('/', 1)
				self.wordset.append(split[0])
				self.tagset.append(split[1])
		self.wordset = list(set(self.wordset))
		self.tagset = list(set(self.tagset))


	# Computes the transition and emission probabilities
	def train(self):
		# Reserve empty string for start/end states
		self.tagset.append("")

		# Initialize count arrays
		transition_counts = {}
		for tag_i in self.tagset:
			transition_counts[tag_i] = {}
			for tag_j in self.tagset:
				transition_counts[tag_i][tag_j] = self.smoothing_value	# Transition smoothing
		emission_counts = {}
		for tag_i in self.tagset:
			emission_counts[tag_i] = {}
			for word_j in self.wordset:
				emission_counts[tag_i][word_j] = 0

		# Count transitions and emissions
		for line in self.lines:
			prev_tag = ""
			for idx, entry in enumerate(line.split()):
				# Get current word and tag indices
				split = entry.rsplit('/', 1)
				word = split[0]
				tag = split[1]
				# Update transition and emission counts
				transition_counts[prev_tag][tag] += 1	# P(tag | prev_tag)
				emission_counts[tag][word] += 1			# P(word | tag)
				prev_tag = tag
			# Empty string for end state
			transition_counts[prev_tag][""] += 1

		# Compute transition probabilities
		for prev_tag in self.tagset:
			total_count = sum(transition_counts[prev_tag].values())	# Number of transitions from prev_tag
			probabilities = {tag : float(count) / total_count for tag, count in transition_counts[prev_tag].items()}
			self.transition_probabilities[prev_tag] = probabilities
		
		# Compute emission probabilities
		for tag in self.tagset:
			if tag == "":
				total_count = 1	# Dummy non-zero count for start state
			else:
				total_count = sum(emission_counts[tag].values())	# Number of occurrences of tag
			probabilities = {word : float(count) / total_count for word, count in emission_counts[tag].items()}
			self.emission_probabilities[tag] = probabilities


	# Writes the final model parameters to file
	def writeModelToFile(self):
		model = {'wordset': self.wordset, 'tagset': self.tagset, 'transitions': self.transition_probabilities, \
			'emissions': self.emission_probabilities}
		with open("hmmmodel.txt", "w", encoding="utf-8") as file_object:
			file_object.write(json.dumps(model, ensure_ascii=False))
			# json.dump(model, file_object, ensure_ascii=False)


def main():
	# Get training data from file
	filename = sys.argv[1]
	with open(filename, "r", encoding="utf-8") as file_object:
		lines = file_object.readlines()

	# Train HMM
	hmm = HMM(lines)
	hmm.parseTrainingData()
	hmm.train()
	hmm.writeModelToFile()


if __name__=="__main__":
	main()