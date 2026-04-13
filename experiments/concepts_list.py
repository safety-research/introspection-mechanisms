# 450 New concepts for experiment 02 (steering evaluation) - avoiding overlap with the 50 baseline concepts
# Baseline concepts to AVOID: Dust, Satellites, Trumpets, Origami, Illusions, Cameras, Lightning,
# Constellations, Treasures, Phones, Trees, Avalanches, Mirrors, Fountains, Quarries, Sadness,
# Xylophones, Secrecy, Oceans, Happiness, Deserts, Kaleidoscopes, Sugar, Vegetables, Poetry,
# Aquariums, Bags, Peace, Caverns, Memories, Frosts, Volcanoes, Boulders, Harmonies, Masquerades,
# Rubber, Plastic, Blood, Amphitheaters, Contraptions, Youths, Dynasties, Snow, Dirigibles,
# Algorithms, Denim, Monoliths, Milk, Bread, Silver
#
# ALSO AVOID (DEFAULT_BASELINE_WORDS): Desks, Jackets, Gondolas, Laughter, Intelligence, Bicycles,
# Chairs, Orchestras, Sand, Pottery, Arrowheads, Jewelry, Daffodils, Plateaus, Estuaries, Quilts,
# Moments, Bamboo, Ravines, Archives, Hieroglyphs, Stars, Clay, Fossils, Wildlife, Flour, Traffic,
# Bubbles, Honey, Geodes, Magnets, Ribbons, Zigzags, Puzzles, Tornadoes, Anthills, Galaxies,
# Poverty, Diamonds, Universes, Vinegar, Nebulae, Knowledge, Marble, Fog, Rivers, Scrolls,
# Silhouettes, Marbles, Cakes, Valleys, Whispers, Pendulums, Towers, Tables, Glaciers, Whirlpools,
# Jungles, Wool, Anger, Ramparts, Flowers, Research, Hammers, Clouds, Justice, Dogs, Butterflies,
# Needles, Fortresses, Bonfires, Skyscrapers, Caravans, Patience, Bacon, Velocities, Smoke,
# Electricity, Sunsets, Anchors, Parchments, Courage, Statues, Oxygen, Time, Fabric, Pasta,
# Snowflakes, Mountains, Echoes, Pianos, Sanctuaries, Abysses, Air, Dewdrops, Gardens, Literature, Rice, Enigmas
#
# ALSO AVOID (hallucination-prone): Apples, Bicycles, Ocean

NEW_CONCEPTS = [
    # ANIMALS (41 concepts)
    "Elephants", "Dolphins", "Eagles", "Serpents", "Wolves", "Koalas", "Octopuses", "Penguins",
    "Crocodiles", "Hummingbirds", "Spiders", "Whales", "Foxes", "Owls", "Jellyfish", "Cheetahs",
    "Salamanders", "Peacocks", "Scorpions", "Gorillas", "Flamingos", "Beetles", "Seahorses", "Ravens",
    "Lobsters", "Chameleons", "Bats", "Kangaroos", "Starfish", "Hedgehogs", "Parrots", "Crabs",
    "Falcons", "Ants", "Squirrels", "Swans", "Turtles", "Bees", "Moose", "Otters",
    "Centipedes",

    # BODY PARTS (22 concepts)
    "Fingers", "Elbows", "Skulls", "Lungs", "Knees", "Spines", "Tongues", "Ankles",
    "Ribs", "Wrists", "Shoulders", "Teeth", "Eyebrows", "Hips", "Necks", "Palms",
    "Earlobes", "Nostrils", "Thumbs", "Foreheads", "Chins", "Heels",

    # COLORS (12 concepts)
    "Crimson", "Turquoise", "Amber", "Violet", "Scarlet", "Indigo", "Maroon", "Magenta",
    "Cerulean", "Burgundy", "Lavender", "Teal",

    # ABSTRACT CONCEPTS (35 concepts)
    "Probability", "Entropy", "Paradoxes", "Symmetry", "Infinity", "Chaos", "Gravity", "Velocity",
    "Momentum", "Friction", "Consciousness", "Wisdom", "Causality", "Integrity", "Curiosity", "Ambition",
    "Duality", "Resilience", "Destiny", "Karma", "Irony", "Satire", "Metaphors", "Analogies",
    "Hypotheses", "Theories", "Proofs", "Axioms", "Dilemmas", "Compromise", "Sacrifice",
    "Redemption", "Transformation", "Evolution", "Revolution",

    # EMOTIONS/MENTAL STATES (25 concepts)
    "Anxiety", "Jealousy", "Gratitude", "Nostalgia", "Frustration", "Excitement", "Boredom",
    "Confusion", "Disgust", "Envy", "Pride", "Shame", "Guilt", "Loneliness", "Contentment",
    "Skepticism", "Wonder", "Awe", "Despair", "Hope", "Regret", "Anticipation", "Dread",
    "Serenity", "Melancholy",

    # FOODS/BEVERAGES (35 concepts)
    "Chocolate", "Cinnamon", "Ginger", "Tofu", "Wasabi", "Mustard", "Pepper", "Garlic",
    "Onions", "Lemons", "Oranges", "Bananas", "Mangoes", "Grapes", "Cherries", "Peaches",
    "Almonds", "Walnuts", "Cashews", "Peanuts", "Coffee", "Tea", "Wine", "Whiskey",
    "Cheese", "Butter", "Eggs", "Caviar", "Sausages", "Dumplings", "Polenta", "Noodles",
    "Soups", "Pizzas", "Sushi",

    # MATERIALS (18 concepts)
    "Copper", "Iron", "Steel", "Titanium", "Aluminum", "Granite", "Kevlar", "Limestone",
    "Concrete", "Glass", "Ceramic", "Porcelain", "Leather", "Silk", "Cotton", "Nylon",
    "Linen", "Velvet",

    # PROFESSIONS (28 concepts)
    "Surgeons", "Architects", "Engineers", "Scientists", "Lawyers", "Pilots", "Chefs", "Farmers",
    "Teachers", "Nurses", "Firefighters", "Detectives", "Journalists", "Astronomers", "Archaeologists", "Philosophers",
    "Economists", "Therapists", "Electricians", "Plumbers", "Carpenters", "Tailors", "Librarians", "Curators",
    "Diplomats", "Translators", "Choreographers", "Sculptors",

    # ACTIONS/VERBS as nouns (28 concepts)
    "Swimming", "Dancing", "Climbing", "Running", "Jumping", "Diving", "Skating", "Skiing",
    "Surfing", "Fishing", "Hunting", "Gardening", "Cooking", "Painting", "Writing", "Reading",
    "Singing", "Whistling", "Laughing", "Crying", "Sleeping", "Dreaming", "Thinking", "Planning",
    "Building", "Creating", "Discovering", "Exploring",

    # SCIENCE TERMS (22 concepts)
    "Photons", "Electrons", "Neutrons", "Protons", "Molecules", "Atoms", "Quarks", "Neurons",
    "Enzymes", "Proteins", "Chromosomes", "Genes", "Viruses", "Bacteria", "Fungi", "Ecosystems",
    "Mitochondria", "Ribosomes", "Comets", "Asteroids", "Isotopes", "Catalysts",

    # EVERYDAY OBJECTS (35 concepts)
    "Scissors", "Stethoscopes", "Screws", "Binoculars", "Threads", "Buttons", "Zippers",
    "Candles", "Lamps", "Clocks", "Keys", "Locks", "Chains", "Ropes", "Ladders",
    "Brushes", "Combs", "Towels", "Blankets", "Pillows", "Curtains", "Carpets", "Chandeliers",
    "Wardrobes", "Shelves", "Drawers", "Boxes", "Jars", "Bottles", "Cups", "Plates",
    "Forks", "Spoons", "Knives", "Pans",

    # BUILDINGS/STRUCTURES (22 concepts)
    "Cathedrals", "Mosques", "Temples", "Pyramids", "Castles", "Cottages", "Lighthouses", "Windmills",
    "Factories", "Warehouses", "Barns", "Bridges", "Tunnels", "Dams", "Pagodas", "Monuments",
    "Stadiums", "Arenas", "Libraries", "Museums", "Hospitals", "Prisons",

    # VEHICLES/TRANSPORT (18 concepts)
    "Rickshaws", "Motorcycles", "Automobiles", "Trains", "Submarines", "Helicopters", "Airplanes", "Rockets",
    "Sailboats", "Canoes", "Kayaks", "Ferries", "Tractors", "Ambulances", "Tanks", "Carriages",
    "Sleds", "Skateboards",

    # SPORTS (18 concepts)
    "Basketball", "Football", "Baseball", "Tennis", "Golf", "Boxing", "Wrestling", "Archery",
    "Fencing", "Gymnastics", "Weightlifting", "Marathon", "Triathlon", "Hockey", "Cricket", "Rugby",
    "Volleyball", "Badminton",

    # MUSIC/ART (18 concepts)
    "Violins", "Cellos", "Bagpipes", "Drums", "Flutes", "Harps", "Guitars", "Saxophones",
    "Clarinets", "Accordions", "Symphonies", "Operas", "Ballets", "Sculptures", "Mosaics", "Tapestries",
    "Frescoes", "Murals",

    # WEATHER (12 concepts)
    "Thunderstorms", "Hurricanes", "Auroras", "Droughts", "Floods", "Hailstorms", "Rainbows", "Squalls",
    "Mist", "Dew", "Sleet", "Blizzards",

    # GEOGRAPHY (12 concepts)
    "Marshes", "Lagoons", "Lakes", "Tundras", "Canyons", "Savannas", "Atolls", "Islands",
    "Peninsulas", "Archipelagos", "Fjords", "Deltas",

    # TIME/TEMPORAL CONCEPTS (12 concepts)
    "Centuries", "Decades", "Millennia", "Equinoxes", "Instants", "Eternities", "Eras", "Epochs",
    "Seasons", "Twilights", "Dawns", "Solstices",

    # SOCIAL CONCEPTS (22 concepts)
    "Weddings", "Funerals", "Ceremonies", "Rituals", "Traditions", "Customs", "Festivals", "Celebrations",
    "Protests", "Elections", "Debates", "Negotiations", "Treaties", "Alliances", "Conflicts", "Reconciliations",
    "Partnerships", "Rivalries", "Friendships", "Mentorships", "Apprenticeships", "Citizenship",

    # PLANTS (15 concepts)
    "Roses", "Tulips", "Orchids", "Sunflowers", "Daisies", "Lilies", "Ferns", "Mosses",
    "Cacti", "Vines", "Shrubs", "Herbs", "Grasses", "Seaweed", "Mushrooms",
]

# Verify count
print(f"Total new concepts: {len(NEW_CONCEPTS)}")
print(f"Unique concepts: {len(set(NEW_CONCEPTS))}")

# Check for any duplicates
from collections import Counter
counts = Counter(NEW_CONCEPTS)
duplicates = [(word, count) for word, count in counts.items() if count > 1]
if duplicates:
    print(f"WARNING - Duplicates found: {duplicates}")
else:
    print("No duplicates - all concepts are unique!")

# Also check overlap with baseline concepts
BASELINE = [
    "Dust", "Satellites", "Trumpets", "Origami", "Illusions", "Cameras", "Lightning",
    "Constellations", "Treasures", "Phones", "Trees", "Avalanches", "Mirrors", "Fountains", "Quarries",
    "Sadness", "Xylophones", "Secrecy", "Oceans", "Happiness", "Deserts", "Kaleidoscopes", "Sugar",
    "Vegetables", "Poetry", "Aquariums", "Bags", "Peace", "Caverns", "Memories", "Frosts",
    "Volcanoes", "Boulders", "Harmonies", "Masquerades", "Rubber", "Plastic", "Blood", "Amphitheaters",
    "Contraptions", "Youths", "Dynasties", "Snow", "Dirigibles", "Algorithms", "Denim", "Monoliths",
    "Milk", "Bread", "Silver"
]

overlap = set(NEW_CONCEPTS) & set(BASELINE)
if overlap:
    print(f"WARNING - Overlap with baseline: {overlap}")
else:
    print("No overlap with baseline concepts - good!")

# Also check overlap with DEFAULT_BASELINE_WORDS (used for vector extraction baseline)
DEFAULT_BASELINE_WORDS = [
    "Desks", "Jackets", "Gondolas", "Laughter", "Intelligence",
    "Bicycles", "Chairs", "Orchestras", "Sand", "Pottery",
    "Arrowheads", "Jewelry", "Daffodils", "Plateaus", "Estuaries",
    "Quilts", "Moments", "Bamboo", "Ravines", "Archives",
    "Hieroglyphs", "Stars", "Clay", "Fossils", "Wildlife",
    "Flour", "Traffic", "Bubbles", "Honey", "Geodes",
    "Magnets", "Ribbons", "Zigzags", "Puzzles", "Tornadoes",
    "Anthills", "Galaxies", "Poverty", "Diamonds", "Universes",
    "Vinegar", "Nebulae", "Knowledge", "Marble", "Fog",
    "Rivers", "Scrolls", "Silhouettes", "Marbles", "Cakes",
    "Valleys", "Whispers", "Pendulums", "Towers", "Tables",
    "Glaciers", "Whirlpools", "Jungles", "Wool", "Anger",
    "Ramparts", "Flowers", "Research", "Hammers", "Clouds",
    "Justice", "Dogs", "Butterflies", "Needles", "Fortresses",
    "Bonfires", "Skyscrapers", "Caravans", "Patience", "Bacon",
    "Velocities", "Smoke", "Electricity", "Sunsets", "Anchors",
    "Parchments", "Courage", "Statues", "Oxygen", "Time",
    "Butterflies", "Fabric", "Pasta", "Snowflakes", "Mountains",
    "Echoes", "Pianos", "Sanctuaries", "Abysses", "Air",
    "Dewdrops", "Gardens", "Literature", "Rice", "Enigmas",
]

overlap_baseline_words = set(NEW_CONCEPTS) & set(DEFAULT_BASELINE_WORDS)
if overlap_baseline_words:
    print(f"WARNING - Overlap with DEFAULT_BASELINE_WORDS: {overlap_baseline_words}")
else:
    print("No overlap with DEFAULT_BASELINE_WORDS - good!")

# Check for hallucination-prone concepts
HALLUCINATION_PRONE = ["Apples", "Bicycles", "Ocean"]
overlap_halluc = set(NEW_CONCEPTS) & set(HALLUCINATION_PRONE)
if overlap_halluc:
    print(f"WARNING - Contains hallucination-prone concepts: {overlap_halluc}")
else:
    print("No hallucination-prone concepts - good!")
