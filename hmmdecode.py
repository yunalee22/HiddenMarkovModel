import sys
import json
import numpy

""" Class to represent a Viterbi algorithm decoder. """
class ViterbiDecoder():

	def __init__(self, lines):
		self.lines = lines
		self.wordset = []
		self.tagset = []
		self.transitions = {}	# Transition probabilities
		self.emissions = {}		# Emission probabilities

	def readModelFromFile(self):
		with open("hmmmodel.txt", "r", encoding="utf-8") as file_object:
			model = json.load(file_object)
			self.wordset = model['wordset']
			self.tagset = model['tagset']
			self.transitions = model['transitions']
			self.emissions = model['emissions']

	def decode(self):
		tag_sequences = []
		output_file = open("hmmoutput.txt", "w", encoding="utf-8")

		# Iterate through sentences		
		for line in self.lines:
			words = line.split()
			# print words
			num_words = len(words)

			# Initialize a new trellis for each sentence
			trellis = []
			backpointers = []

			# Start state
			trellis.append({})
			backpointers.append({})
			for tag in self.tagset:
				if words[0] in self.emissions[tag]:	# Known word
					trellis[0][tag] = self.transitions[""][tag] * self.emissions[tag][words[0]]
				else:	# Unseen word
					trellis[0][tag] = self.transitions[""][tag]
				backpointers[0][tag] = ""

			# Decode observations
			time_step = 1
			for word in words[1:]:
				trellis.append({})
				backpointers.append({})
				for tag in self.tagset:
					# Take max probability
					max_probability = 0.0
					max_prev_tag = ""
					for prev_tag in self.tagset:
						if word in self.emissions[tag]:	# Known word
							probability = trellis[time_step - 1][prev_tag] * \
								self.transitions[prev_tag][tag] * self.emissions[tag][word]
						else:	# Unseen word
							probability = trellis[time_step - 1][prev_tag] * self.transitions[prev_tag][tag]
						if probability > max_probability:
							max_probability = probability
							max_prev_tag = prev_tag
					# Update trellis and backpointers
					trellis[time_step][tag] = max_probability
					backpointers[time_step][tag] = max_prev_tag
				time_step += 1	# Increment time step
						
			# End state
			trellis.append({})
			backpointers.append({})
			max_probability = 0.0
			max_prev_tag = ""
			for prev_tag in self.tagset:
				probability = trellis[time_step - 1][prev_tag] * self.transitions[prev_tag][""]
				if probability > max_probability:
					max_probability = probability
					max_prev_tag = prev_tag
			# Update trellis and backpointers
			trellis[time_step][""] = max_probability
			backpointers[time_step][""] = max_prev_tag

			# Write tag sequence to file
			tag_sequence = []
			curr_tag = ""
			while time_step != 0:
				backpointer = backpointers[time_step][curr_tag]
				tag_sequence.insert(0, backpointer)
				curr_tag = backpointer
				time_step -= 1
			tag_sequences.append(tag_sequence)

			# Write sequence to file
			for i in range(num_words):
				string = words[i] + "/" + tag_sequence[i] + " "
				output_file.write(string)
			output_file.write("\n")

		return tag_sequences


def main():
	# Get test data from file
	filename = sys.argv[1]
	with open(filename) as file_object:
		lines = file_object.readlines()

	# Test HMM
	decoder = ViterbiDecoder(lines)
	decoder.readModelFromFile()
	tag_sequence = decoder.decode()


if __name__=="__main__":
	main()