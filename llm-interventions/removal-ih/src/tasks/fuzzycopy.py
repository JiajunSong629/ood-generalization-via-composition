import numpy as np
import sys
from collections import Counter
import random

import json

# Function to load JSON data from a file
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path, indent = 4):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent = indent)

import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))


################################################################################################### Upper
# Construct the relative path to the 'factory/classnames.txt'
file_path = os.path.join(current_dir, 'factory', 'classnames.txt')

def read_txt(file_path):
    long = []
    # Reading the file and converting each line into an item in a list
    with open(file_path, 'r') as file:
        for line in file.readlines():
            name = line[10:].split("\n")[0].lower().replace("(","").replace(")","").replace("/","")
            name = name.split(" ")
            name = [i for i in name if i != ""]
            long += name
            if "" in name:
                print(name)
            
            
        retlist = sorted(set(long))

    return retlist

NAMES = read_txt(file_path)

CIFAR_NAMES = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm"
]


################################################################################################ verb
# Correcting the error and completing the list
VERB_PAIRS = [
    ("accept", "accepted"), ("act", "acted"), ("ask", "asked"), ("bake", "baked"), ("begin", "began"),
    ("break", "broke"), ("bring", "brought"), ("build", "built"), ("buy", "bought"), ("catch", "caught"),
    ("choose", "chose"), ("come", "came"), ("cut", "cut"), ("do", "did"), ("draw", "drew"),
    ("drink", "drank"), ("drive", "drove"), ("eat", "ate"), ("fall", "fell"), ("feel", "felt"),
    ("fight", "fought"), ("find", "found"), ("fly", "flew"), ("forget", "forgot"), ("forgive", "forgave"),
    ("freeze", "froze"), ("get", "got"), ("give", "gave"), ("go", "went"), ("grow", "grew"),
    ("hang", "hung"), ("have", "had"), ("hear", "heard"), ("hide", "hid"), ("hit", "hit"),
    ("hold", "held"), ("hurt", "hurt"), ("keep", "kept"), ("know", "knew"), ("lay", "laid"),
    ("lead", "led"), ("leave", "left"), ("lend", "lent"), ("let", "let"), ("lie", "lay"),
    ("light", "lit"), ("lose", "lost"), ("make", "made"), ("mean", "meant"), ("meet", "met"),
    ("pay", "paid"), ("put", "put"), ("read", "read"), ("ride", "rode"), ("ring", "rang"),
    ("rise", "rose"), ("run", "ran"), ("say", "said"), ("see", "saw"), ("sell", "sold"),
    ("send", "sent"), ("set", "set"), ("shake", "shook"), ("shine", "shone"), ("shoot", "shot"),
    ("show", "showed"), ("shut", "shut"), ("sing", "sang"), ("sit", "sat"), ("sleep", "slept"),
    ("speak", "spoke"), ("spend", "spent"), ("stand", "stood"), ("steal", "stole"), ("stick", "stuck"),
    ("strike", "struck"), ("swear", "swore"), ("sweep", "swept"), ("swim", "swam"), ("take", "took"),
    ("teach", "taught"), ("tear", "tore"), ("tell", "told"), ("think", "thought"), ("throw", "threw"),
    ("understand", "understood"), ("wake", "woke"), ("wear", "wore"), ("win", "won"), ("write", "wrote"),

    ("achieve", "achieved"), ("advise", "advised"), ("agree", "agreed"), ("answer", "answered"), ("appear", "appeared"),
    ("apply", "applied"), ("argue", "argued"), ("arrange", "arranged"), ("arrive", "arrived"), ("ask", "asked"),
    ("attach", "attached"), ("attempt", "attempted"), ("attend", "attended"), ("avoid", "avoided"), ("be", "was/were"),
    ("beat", "beat"), ("become", "became"), ("begin", "began"), ("believe", "believed"), ("borrow", "borrowed"),
    ("break", "broke"), ("breathe", "breathed"), ("bring", "brought"), ("build", "built"), ("burn", "burned"),
    ("buy", "bought"), ("calculate", "calculated"), ("call", "called"), ("can", "could"), ("care", "cared"),
    ("carry", "carried"), ("catch", "caught"), ("cause", "caused"), ("change", "changed"), ("charge", "charged"),
    ("check", "checked"), ("choose", "chose"), ("claim", "claimed"), ("clean", "cleaned"), ("clear", "cleared"),
    ("climb", "climbed"), ("close", "closed"), ("collect", "collected"), ("come", "came"), ("compare", "compared"),
    ("compete", "competed"), ("complain", "complained"), ("complete", "completed"), ("concern", "concerned"), ("confirm", "confirmed"),
    ("connect", "connected"), ("consider", "considered"), ("consult", "consulted"), ("contain", "contained"), ("continue", "continued"),
    ("contribute", "contributed"), ("control", "controlled"), ("cook", "cooked"), ("copy", "copied"), ("correct", "corrected"),
    
    ("cost", "cost"), ("count", "counted"), ("cover", "covered"), ("create", "created"), ("cry", "cried"),
    ("cut", "cut"), ("damage", "damaged"), ("dance", "danced"), ("deal", "dealt"), ("decide", "decided"),
    ("deliver", "delivered"), ("demand", "demanded"), ("deny", "denied"), ("depend", "depended"), ("describe", "described"),
    ("deserve", "deserved"), ("destroy", "destroyed"), ("determine", "determined"), ("develop", "developed"), ("die", "died"),
    ("disagree", "disagreed"), ("discover", "discovered"), ("discuss", "discussed"), ("dislike", "disliked"), ("divide", "divided"),
    ("do", "did"), ("draw", "drew"), ("dream", "dreamed"), ("drive", "drove"), ("drop", "dropped"),
    ("earn", "earned"), ("eat", "ate"), ("emphasize", "emphasized"), ("enable", "enabled"), ("encourage", "encouraged"),
    ("engage", "engaged"), ("enhance", "enhanced"), ("enjoy", "enjoyed"), ("ensure", "ensured"), ("enter", "entered"),
    ("establish", "established"), ("examine", "examined"), ("exist", "existed"), ("expand", "expanded"), ("expect", "expected"),
    ("experience", "experienced"), ("explain", "explained"), ("explore", "explored"), ("express", "expressed"), ("extend", "extended"),
    ("face", "faced"), ("fail", "failed"), ("fall", "fell"), ("feed", "fed"), ("feel", "felt"),
    ("fight", "fought"), ("find", "found"), ("finish", "finished"), ("fit", "fit/fitted"), ("fly", "flew"),
    ("focus", "focused"), ("follow", "followed"), ("forbid", "forbade"), ("forget", "forgot"), ("forgive", "forgave"),
    ("freeze", "froze"), ("get", "got"), ("give", "gave"), ("go", "went"), ("grow", "grew"),
    ("handle", "handled"), ("hang", "hung"), ("happen", "happened"), ("hate", "hated"), ("have", "had"),
    ("hear", "heard"), ("help", "helped"), ("hide", "hid"), ("hit", "hit"), ("hold", "held"),
    ("hope", "hoped"), ("hurt", "hurt"), ("identify", "identified"), ("ignore", "ignored"), ("imagine", "imagined"),
    ("implement", "implemented"), ("imply", "implied"), ("improve", "improved"), ("include", "included"), ("incorporate", "incorporated"),
    ("indicate", "indicated"), ("inform", "informed"), ("insist", "insisted"), ("install", "installed"), ("intend", "intended"),
    ("introduce", "introduced"), ("invest", "invested"), ("investigate", "investigated"), ("involve", "involved"), ("join", "joined"),
    ("jump", "jumped"), ("justify", "justified"), ("keep", "kept"), ("kick", "kicked"), ("kill", "killed"),
    ("kiss", "kissed"), ("knock", "knocked"), ("know", "knew"), ("lack", "lacked"), ("laugh", "laughed"),
    ("lay", "laid"), ("lead", "led"), ("lean", "leaned"), ("learn", "learned"), ("leave", "left"),
    ("lend", "lent"), ("let", "let"), ("lie", "lay"), ("like", "liked"), ("listen", "listened"),
    ("live", "lived"), ("look", "looked"), ("lose", "lost"), ("love", "loved"), ("make", "made"),
    ("manage", "managed"), ("mark", "marked"), ("matter", "mattered"), ("mean", "meant"), ("measure", "measured"),
    ("meet", "met"), ("mention", "mentioned"), ("mind", "minded"), ("miss", "missed"), ("mix", "mixed"),
    ("move", "moved"), ("need", "needed"), ("negotiate", "negotiated"),

    ("observe", "observed"), ("obtain", "obtained"), ("occur", "occurred"), ("offer", "offered"), ("open", "opened"),
    ("operate", "operated"), ("order", "ordered"), ("organize", "organized"), ("overcome", "overcame"), ("participate", "participated"),
    ("pay", "paid"), ("perform", "performed"), ("persuade", "persuaded"), ("place", "placed"), ("plan", "planned"),
    ("play", "played"), ("point", "pointed"), ("prefer", "preferred"), ("prepare", "prepared"), ("present", "presented"),
    ("prevent", "prevented"), ("produce", "produced"), ("promise", "promised"), ("promote", "promoted"), ("protect", "protected"),
    ("prove", "proved"), ("provide", "provided"), ("publish", "published"), ("pull", "pulled"), ("push", "pushed"),
    ("put", "put"), ("question", "questioned"), ("quit", "quit"), ("reach", "reached"), ("read", "read"),
    ("realize", "realized"), ("receive", "received"), ("recognize", "recognized"), ("recommend", "recommended"), ("record", "recorded"),
]


def generator(n_examples: int = 10, n_test: int = 5, setting = "upper", template: callable = None):
    while True:
        if setting == "upper":
            seq_names = random.choices(CIFAR_NAMES, k=n_examples)
            test_names = [name.upper() for name in seq_names]
        elif setting == "past":
            pairs = random.choices(VERB_PAIRS, k=n_examples)
            seq_names = [pair[0] for pair in pairs]
            test_names = [pair[1] for pair in pairs]

        prompt = " ".join(seq_names + test_names[:n_test])
        answer = " ".join(test_names[n_test:])

        yield prompt, answer


def main():
    #################################### classnames
    
    # names = read_txt(file_path)
    names = CIFAR_NAMES
    print("total names: ", len(names))
    print("names: ", names[:5])
    print("type names: ", type(names))

    verb_pairs = VERB_PAIRS
    print("total verbs: ", len(verb_pairs))
    print("verb_pairs: ", verb_pairs[:5])
    print("type verb_pairs: ", type(verb_pairs))

    pairs = random.choices(VERB_PAIRS, k=4)
    print(pairs)


if __name__ == "__main__":
    main()
