import random

### Output Messages ###
no_runs_found_message_func = lambda prefix: "No run logs found. If this is unexpected, try the following steps in order: \n 1) Double check that the name of the folder containing the event file for this question contains the string(s): " + str(prefix) + "\n 2) Rewatch the instructional video on Piazza explaining how to correctly submit assignments. \n 3) Ask for assistance on Piazza, staff is here to help!"
corruped_error_message = "Your event file is corrupted. This is likely caused by an attempt to change the logging process in some way: \n\n"
no_ef_found_message_func = lambda key: "No logger data found. Try the following steps in order: \n 1) Check that the folder for this question contains an event file. \n 2) If there is an event file, double check that it contains data for the key: {0} \n 3) Ask for assistance on Piazza, staff is here to help!".format(key)
low_score_message_func = lambda thresh: "You must reach a score of {0} to receive full credit for this question.".format(thresh)
unknown_error_message = "Unknown Error. Please contact course staff for assistance in debugging this: \n\n"
full_credit_message = "You received full credit for this question!"

### Joke Lists ###
cs_jokes = [
"Two bytes meet. The first byte asks, 'Are you ill?' The second byte replies, 'No, just feeling a bit off.'",
"Q. How did the programmer die in the shower? \n A. He read the shampoo bottle instructions: Lather. Rinse. Repeat.",
"How many programmers does it take to change a light bulb? None, its a hardware problem",
"Eight bytes walk into a bar.  The bartender asks, 'Can I get you anything?' 'Yeah', reply the bytes. 'Make us a double.'",
"There are only 10 kinds of people in this world: those who know binary and those who don't.",
"A programmer walks to the butcher shop and buys a kilo of meat.  An hour later he comes back upset that the butcher shortchanged him by 24 grams.",
"Theory is when you know everything but nothing works. Practice is when everything works and no one knows why. Here at CS285, practice and theory are combined: nothing works and no one knows why.",
"There are three kinds of lies: Lies, damned lies, and paper benchmarks.",
"All programmers are playwrights, and all computers are lousy actors.",
"Have you heard about the new Cray super computer?  It's so fast, it executes an infinite loop in 6 seconds.",
"The generation of random numbers is too important to be left to chance.",
"I just saw my life flash before my eyes and all I could see was a close tag...",
"The computer is mightier than the pen, the sword, and usually, the programmer.",
"Debugging: Searching for hay in a needle stack.",
"One hundred little bugs in the code, One hundred little bugs. Fix a bug, link the fix in, One hundred little bugs in the code.",
]

jokes = [
"Santa Claus' helpers are known as subordinate Clauses.",
"She had a photographic memory but never developed it.",
"The two pianists had a good marriage. They always were in a chord.",
"I was struggling to figure out how lightning works, but then it struck me.",
"The grammarian was very logical. He had a lot of comma sense.",
"A chicken farmer's favorite car is a coupe.",
"What do you call a person rabid with wordplay? An energizer punny.",
"I've been to the dentist many times so I know the drill.",
"The other day I held the door open for a clown. I thought it was a nice jester.",
"A chicken crossing the road is truly poultry in motion.",
"The politician is not one for Indian food. But he's good at currying favors.",
"How do construction workers party? They raise the roof.",
"A boiled egg every morning is hard to beat.",
"After hours of waiting for the bowling alley to open, we finally got the ball rolling.",
"When a Sarah returned her new gown, she got post traumatic dress syndrome.",
"What do you call a bear with no teeth? A gummy bear.",
"Two antennas met on a roof, fell in love and got married. The ceremony wasn't much, but the reception was brilliant!",
"Always trust a glue salesman. They tend to stick to their word.",
"I thought Santa was going to be late, but he arrived in the Nick of time.",
"Every calendar's days are numbered.",
"A bicycle can't stand on its own because it is two tired.",
"No matter how much you push the envelope, it will still be stationery.",
"A dog gave birth to puppies near the road and was cited for littering.",
"A pessimist's blood type is always B negative.",
"I went to a seafood disco last week... and pulled a mussel.",
"Two peanuts walk into a bar, and one was a salted.",
"Reading while sunbathing makes you well red.",
]

### Joke Messages ###
cs_joke_message = "As a special reward for getting full credit, here's a CS joke: \n\n" + random.choice(cs_jokes)
joke_message = "Don't stop training those agents! Here's a joke to keep you going: \n\n" + random.choice(jokes)

