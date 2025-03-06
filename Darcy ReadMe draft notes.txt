At first I tried, NER, via Label Studio. It seemed unlikely to work so much better than feeding sections of the book to ChatGPT that it would be worth the extra time manually parsing its labels. In addition to needing parts of the text highlighted, often for <darcy-actions> and <others-impressions-of-darcy> I would need them interpreted as well. A GPT model could both process and interpret simultaneously. The data still needed to be reviewed manually, but this saved much more time.


Many of Wickham's false representations of Darcy have been excluded, except those that show a more broad representation of the general view of him.

In the interest of preventing it from memorizing the plot of the book so much as simply Darcy's speaking patterns, I broke up each bit of dialogue into individual sentences unless I felt context was necessary.

It's possibly a mistake to break up the dialogue, because the AI will not understand that the shorter sentences are part of longer discourse. It make learn that Darcy is always as brief as possible, never capable of warm and open discussion.

I kept the data in story order as much as possible, in the hope GPT-2 could use its previous 'familiarity' with the story to overcome any poorly given data.

In considering how to train the model, I needed to decide how to present the data. My instinct was to present the model with all dialogue at once, then all actions at once, and then all impressions. This would make the most sense as presentation in a text file, certainly. However, I've decided to present dialogue and actions in order, and then impressions sprinkled in at random halfway through. I don't want to overtrain it on early impressions of Darcy, given their general innacuracy. However, there is the fact of darcy's own dialogue becoming warmer toward the end. It might be good to train it on some of that early on as well, lest it seem too cold.

So I will consider training twice. Once on dialogue and actions in order, then on dialogue and actions in random order, still with impressions only halfway through.

I had to give it questions of a more general nature, to prevent DarcyGPT from being purely an 1800s gent.

I attempted to apply my prompt for CoPilot, but it didn't perform well there. I had tuned it to get strong, consistent results from ChatGPT and it didn't appear to cross over very well.

There was some difficulty involved in the creation of the training questions. If these didn't sound perfectly like Darcy, all of my hard work in creating training data tuned to his personality could be diluted by a non-Darcy voice. The generalization inherent in the output of a GPT model make this difficult enough already. I created a script to extract all dialogue from the training data, and used selections to few-shot train instances of ChatGPT to sound as much like Darcy as possible before having it interpret its own answers to these questions into Darcy's voice. This ensured that DarcyGPT would adopt the speech patterns of Darcy, but would also maintain generally the same neutral, inoffensive views of the corporate AI model forming its foundation.

It was important to see if a GPT model could identify the purpose of each category of data consistently, to test whether I'd completed my processing successfully. If it could, for instance, read a small group of others' impressions of Darcy and understand implicitly that they were others' impressions of Darcy, then I could consider my data categorization a success.

When writing the synthesized answers I would consult a file containing all of Darcy's spoken and written dialogue, looking for words he would use.

Why this project, in particular? AS I read about AI, it struck me how difficult it would be to give it a personality of any kind, given the sheer volume of input required in order to perfectly complete complicated tasks. It seemed to me that there weren't enough examples of character dialogue in any work of fiction to truly make mimicry of a personality possible. I recalled that every time I'd asked a GPT model to produce text in a particular voice, however fun the results were, they were perceivably imperfect. I became curious to see if the fine-tuning process built specifically around the strong personality of a strong writer's character (such as that of Mr. Darcy) could improve the results that one-or-few-shot training of a complete model made so incomplete.

I had to ensure that aspects of Darcy's personality which would clash with the GPT model's modern speech patterns–his propensity to say "men" instead of the more modern "men and women," for instance–would not increase perplexity in the model's responses.

Insurmountable difficulties: Darcy is a person, and would have opinions. A GPT model cannot have opinions. Ask any GPT model its favorite film. They have none. Darcy, as a person, fictional or otherwise, would have opinions, favorite books, plays. However, it's possible that without training a GPT model from scratch in the voice of 19th-century tastes, it's impossible for him to have favorites of anything.


There is no possibility of training using an evaluation dataset, as there's already so little data in Darcy's voice to begin with that I can't remove any data to use for testing during training. Instead, I've turned off evaluation_strategy and am hoping for the best after training. Even a 90/10 split on evaluation is impractical, because it makes the eval set so small that there's no way to effectively test.

I'm having it parrot my questinos back at me before adding anything of its own in reply. What it does add sounds more and more like Darcy, but parroting is not normal behavior. I think the culprit may be twofold.
	1. There's too much categorical variety in the data, and not enough data for the model to recognize the significance of that variety.
	2. It seems not to know the difference between the questions and its own dialogue.
I'll train it one more time, one epoch only. Then, if that doesn't work, I'll remove all but the dialogue and questions, and add prompts for each of Darcy's dialogue selections. All 340!



Problems! I had a role in teaching Darcy what to think. How to decide between different prompt constructions? Do I want to overfit the model's ability to think about relationships and seduction, or take an opportunity to introduce some topics that might otherwise be appropriate but less talked of?
<user> What do you think about the arts people employ to influence one another?
<user> What are 
<user> What do you think about the arts ladies employ to captivate potential lovers?
<darcy-dialogue> Whatever bears affinity to cunning is despicable.  



I had to remove the madams, and miss bennnets from the text where there was no madam or miss bennet being explicitly referred to in context, or the model may have considered them an essential part of responding to prompts, no matter the context.


GPT2medium 335M Params
GPT2LArge 774M Params
GPT-J 6B


