import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def generate_question(prompt, max_length=150):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract the question part from the generated text
    question = question.replace(prompt, '').strip()
    return question

def provide_feedback(answer, context, max_length=150):
    feedback_prompt = f"Answer: {answer}\nContext: {context}\nProvide feedback on the answer."
    input_ids = tokenizer.encode(feedback_prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
    feedback = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return feedback

def main():
    context = (
        "You are interviewing a candidate for a DevOps Engineer position at a well-established Fortune 500 company. "
        "The ideal candidate should have experience developing and executing multi-cloud infra, experience in Kubernetes, strong analytical skills, "
        "and the ability to collaborate effectively with cross-functional teams. They should also be passionate about staying up-to-date with the latest DevOps Practices trends and technologies."
    )

    print("Welcome to the mock interview!")
    print("Context:", context)

    for i in range(5):  # Conduct 5 questions in the mock interview
        question_prompt = (
            f"Generate a thoughtful, open-ended interview question for a candidate for the DevOps Engineer position. "
            f"Context: {context}\nQuestion: "
        )
        question = generate_question(question_prompt)
        print(f"\nQuestion {i+1}: {question}")

        answer = input("Your answer: ")
        feedback = provide_feedback(answer, context)
        print(f"Feedback: {feedback}")

if __name__ == "__main__":
    main()
