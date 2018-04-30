from distutils.core import setup

setup(
	name='SiameseModel',
	version='0.0',
	packages=['duplicate_questions', 
			  'duplicate_questions.models', 
			  'duplicate_questions.models.siamese_bilstm',
			  'duplicate_questions.data',
			  'duplicate_questions.data.instances',
			  'duplicate_questions.data.tokenizers',
			  'duplicate_questions.util'],
)