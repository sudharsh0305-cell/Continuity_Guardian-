import os
import re

class ScriptParser:
    def __init__(self):
        pass

    def parse_text_script(self, text_content):
        """
        Parses industry standard screenplays into structured scenes.
        Looks for INT. / EXT. sluglines.
        Identifies character dialogues, actions, props, and wardrobe.
        """
        scenes = []
        current_scene = None
        scene_counter = 0
        
        lines = text_content.splitlines()
        
        # Regex patterns
        slugline_pattern = re.compile(r'^(INT\.|EXT\.)\s+(.+)$', re.IGNORECASE)
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
                
            # Check for slugline
            slug_match = slugline_pattern.match(line_stripped)
            if slug_match:
                if current_scene:
                    scenes.append(current_scene)
                scene_counter += 1
                current_scene = {
                    "scene_number": scene_counter,
                    "slugline": line_stripped,
                    "characters": set(),
                    "wardrobe_notes": [],
                    "props": set(),
                    "content": []
                }
                continue
                
            if current_scene is not None:
                current_scene["content"].append(line_stripped)
                
                # Extract characters (commonly indented or UPPERCASE before dialogue)
                if line_stripped.isupper() and len(line_stripped) < 25 and not line_stripped.startswith("INT.") and not line_stripped.startswith("EXT."):
                    current_scene["characters"].add(line_stripped)
                
                # Heuristics for wardrobe and props
                wardrobe_words = ["wearing", "jacket", "shirt", "hat", "glasses", "suit", "pants", "dress", "coat", "tie"]
                line_lower = line_stripped.lower()
                for word in wardrobe_words:
                    if word in line_lower:
                        current_scene["wardrobe_notes"].append(line_stripped)
                        break
                        
                # Props: objects like "phone", "bag", "cup", "laptop", "ball", "frisbee", "book", "knife", "gun", "keys"
                prop_words = ["phone", "bag", "cup", "laptop", "ball", "frisbee", "book", "knife", "gun", "keys", "briefcase", "glass", "bench", "horse"]
                for p_word in prop_words:
                    if re.search(r'\b' + re.escape(p_word) + r'\b', line_lower):
                        current_scene["props"].add(p_word)
                        
        if current_scene:
            scenes.append(current_scene)
            
        # Convert sets to lists for JSON serialization
        for sc in scenes:
            sc["characters"] = sorted(list(sc["characters"]))
            sc["props"] = sorted(list(sc["props"]))
            
        return scenes

    def parse_script_file(self, file_path):
        if not os.path.exists(file_path):
            print(f"Script file {file_path} not found.")
            return []
            
        _, ext = os.path.splitext(file_path.lower())
        if ext == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return self.parse_text_script(text)
            except ImportError:
                print("[Script Parser] pypdf library not found. Please install pypdf or provide a .txt script.")
                return []
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return self.parse_text_script(text)
