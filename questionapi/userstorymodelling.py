# creating personas list
functions = ['Accountant','Accounts','Accounts Receivables','RTR','R2R','Auditor','Internal auditor','Risk','Finance','Chief Financial Officer','CFO']
designations = ['Executive','Manager','Clerk','Coordinator','Supervisor','Bookkeeper','Specialist']

personas = []

for func in functions:
    for desg in designations:
        personas.append(func+' '+desg)

# creating actions list
actions = ['see real-time information','run an automated report','Run an integrated report','generate a list','create a table','have an overview']
informations = ['payments received','payments from customers','payments that are overdue','payments that are late','payments that are delinquent','customers that are overdue','accounts receivables','all receivables','outstanding payments']

perform_action =[]

for act in actions:
    for info in informations:
        perform_action.append(act+' of '+info)

#goals list
goals = ['check that payments have been made','update my ledger','update my books','follow up on late payments','alert the collections team','send reminders to clients','identify risky accounts','plan for next quarter']

# getting BERT embedding
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')

# cosine similairty
from sklearn.metrics.pairwise import cosine_similarity

# embedding all in the intilaization
personas_embedding = model.encode(personas)
perform_action_embedding = model.encode(perform_action)
goals_embedding = model.encode(goals)

def Q1BERTEncodingCosine(input):

    #embedding the different parts of input
    personas_input_embedding = model.encode([input[0]])
    perform_action_input_embedding = model.encode([input[1]])
    goals_input_embedding = model.encode([input[2]])

    # cosine similarity
    personas_matching = cosine_similarity(personas_input_embedding,personas_embedding)
    perform_action_matching = cosine_similarity(perform_action_input_embedding,perform_action_embedding)
    goals_matching = cosine_similarity(goals_input_embedding,goals_embedding)

    # taking decisions on the basis if similarity
    persona_match = perform_action_match = goals_matching_match = False
    if max(personas_matching[0])>=.9:
        persona_match = True
    if max(perform_action_matching[0])>=.9:
        perform_action_match = True
    if max(goals_matching[0])>=.9:
        goals_matching_match = True

    return [persona_match,perform_action_match,goals_matching_match]
