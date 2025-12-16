#!/usr/bin/env python3
"""
MIDI to ABC Converter
Converts MIDI files to ABC notation using music21.
"""

import warnings
import io
import contextlib
from pathlib import Path
from typing import Optional

# Suppress warnings
warnings.filterwarnings('ignore')

import music21
music21.environment.UserSettings()['warnings'] = 0


class MIDIToABCConverter:
    """Convert MIDI files to ABC notation using music21."""
    
    def __init__(self):
        self.conversion_stats = {'success': 0, 'failed': 0, 'errors': []}
    
    def convert_midi_to_abc(self, midi_path: Path) -> Optional[str]:
        """Convert a MIDI file to ABC notation."""
        try:
            null_stderr = io.StringIO()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with contextlib.redirect_stderr(null_stderr):
                    score = music21.converter.parse(str(midi_path))
            
            abc_str = self._score_to_abc_manual(score)
            
            if abc_str:
                abc_str = self._clean_abc(abc_str)
                if len(abc_str.strip()) > 0:
                    self.conversion_stats['success'] += 1
                    return abc_str
            
            self.conversion_stats['failed'] += 1
            return None
        except Exception as e:
            self.conversion_stats['failed'] += 1
            self.conversion_stats['errors'].append(str(e))
            return None
    
    def _score_to_abc_manual(self, score) -> str:
        """Manually convert music21 score to ABC notation."""
        try:
            abc_lines = []
            
            # ABC header
            abc_lines.append("X:1")
            abc_lines.append("M:4/4")  # Default time signature
            abc_lines.append("L:1/8")  # Default note length
            abc_lines.append("K:C")     # Default key
            
            # Extract time signature if available
            for ts in score.flat.getElementsByClass('TimeSignature'):
                if ts.numerator and ts.denominator:
                    abc_lines[1] = f"M:{ts.numerator}/{ts.denominator}"
                    break
            
            # Extract key signature if available
            try:
                key = score.analyze('key')
                if key:
                    key_name = key.tonic.name
                    mode = 'maj' if key.mode == 'major' else 'min'
                    abc_lines[3] = f"K:{key_name}{mode[0]}"
            except:
                pass
            
            # Convert notes to ABC body
            abc_body = []
            measure_count = 0
            
            for element in score.flat.notesAndRests:
                if isinstance(element, music21.note.Note):
                    abc_body.append(self._note_to_abc(element))
                elif isinstance(element, music21.note.Rest):
                    dur = self._duration_to_abc(element.duration.quarterLength)
                    abc_body.append("z" + dur)
                elif isinstance(element, music21.chord.Chord):
                    # Handle chords (simplified: use first note)
                    if len(element.notes) > 0:
                        abc_body.append(self._note_to_abc(element.notes[0]))
                
                # Add bar lines periodically
                measure_count += 1
                if measure_count % 4 == 0:
                    abc_body.append("|")
            
            body_str = "".join(abc_body)
            if len(body_str) > 80:
                parts = body_str.split("|")
                formatted_parts = []
                for part in parts:
                    if len(part) > 80:
                        words = part.split()
                        line = []
                        for word in words:
                            if len(" ".join(line + [word])) > 80 and line:
                                formatted_parts.append(" ".join(line))
                                line = [word]
                            else:
                                line.append(word)
                        if line:
                            formatted_parts.append(" ".join(line))
                    else:
                        formatted_parts.append(part)
                body_str = "|".join(formatted_parts)
            
            abc_lines.append(body_str)
            return "\n".join(abc_lines) if abc_lines else ""
        except Exception as e:
            return ""
    
    def _note_to_abc(self, note) -> str:
        """Convert a music21 note to ABC notation."""
        try:
            note_name = note.pitch.name[0]
            
            if note.pitch.accidental:
                if note.pitch.accidental.alter == 1:
                    note_name = "^" + note_name
                elif note.pitch.accidental.alter == -1:
                    note_name = "_" + note_name
            
            octave = note.pitch.octave
            if octave < 4:
                note_name = note_name.lower() * (4 - octave)
            elif octave > 4:
                note_name = note_name + "'" * (octave - 4)
            
            dur = self._duration_to_abc(note.duration.quarterLength)
            return note_name + dur
        except Exception:
            return ""
    
    def _duration_to_abc(self, quarter_length: float) -> str:
        """Convert duration in quarter notes to ABC notation."""
        eighth_notes = quarter_length * 2
        eighth_notes = round(eighth_notes * 8) / 8
        
        if eighth_notes <= 0:
            return ""
        elif eighth_notes == 0.5:
            return "/"
        elif eighth_notes == 1.0:
            return ""
        elif eighth_notes == 2.0:
            return "2"
        elif eighth_notes == 3.0:
            return "3"
        elif eighth_notes == 4.0:
            return "4"
        elif eighth_notes == 6.0:
            return "6"
        elif eighth_notes == 8.0:
            return "8"
        else:
            dur_int = int(eighth_notes)
            if dur_int > 0 and dur_int <= 16:
                return str(dur_int)
            else:
                return f"/{int(1/eighth_notes)}" if eighth_notes < 1 else str(int(eighth_notes))
    
    def _clean_abc(self, abc_str: str) -> str:
        """Clean and normalize ABC notation string."""
        lines = abc_str.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('%'):
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)


def convert_single_midi_worker(args):
    """
    Convert a single MIDI file to ABC (for parallel processing).
    This function is completely self-contained to avoid pickling issues in notebooks.
    """
    midi_file, output_dir = args
    
    try:
        # Import all necessary modules in worker process
        import warnings
        import io
        import contextlib
        from pathlib import Path
        import music21
        
        # Suppress warnings in worker process
        warnings.filterwarnings('ignore')
        music21.environment.UserSettings()['warnings'] = 0
        
        # Helper function: duration to ABC
        def duration_to_abc(quarter_length):
            eighth_notes = quarter_length * 2
            eighth_notes = round(eighth_notes * 8) / 8
            if eighth_notes <= 0:
                return ""
            elif eighth_notes == 0.5:
                return "/"
            elif eighth_notes == 1.0:
                return ""
            elif eighth_notes == 2.0:
                return "2"
            elif eighth_notes == 3.0:
                return "3"
            elif eighth_notes == 4.0:
                return "4"
            elif eighth_notes == 6.0:
                return "6"
            elif eighth_notes == 8.0:
                return "8"
            else:
                dur_int = int(eighth_notes)
                if dur_int > 0 and dur_int <= 16:
                    return str(dur_int)
                else:
                    return f"/{int(1/eighth_notes)}" if eighth_notes < 1 else str(int(eighth_notes))
        
        # Helper function: note to ABC
        def note_to_abc(note):
            try:
                note_name = note.pitch.name[0]
                if note.pitch.accidental:
                    if note.pitch.accidental.alter == 1:
                        note_name = "^" + note_name
                    elif note.pitch.accidental.alter == -1:
                        note_name = "_" + note_name
                octave = note.pitch.octave
                if octave < 4:
                    note_name = note_name.lower() * (4 - octave)
                elif octave > 4:
                    note_name = note_name + "'" * (octave - 4)
                dur = duration_to_abc(note.duration.quarterLength)
                return note_name + dur
            except Exception:
                return ""
        
        # Parse MIDI file
        null_stderr = io.StringIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stderr(null_stderr):
                score = music21.converter.parse(str(midi_file))
        
        # Convert to ABC manually
        abc_lines = []
        abc_lines.append("X:1")
        abc_lines.append("M:4/4")
        abc_lines.append("L:1/8")
        abc_lines.append("K:C")
        
        # Extract time signature if available
        for ts in score.flat.getElementsByClass('TimeSignature'):
            if ts.numerator and ts.denominator:
                abc_lines[1] = f"M:{ts.numerator}/{ts.denominator}"
                break
        
        # Extract key signature if available
        try:
            key = score.analyze('key')
            if key:
                key_name = key.tonic.name
                mode = 'maj' if key.mode == 'major' else 'min'
                abc_lines[3] = f"K:{key_name}{mode[0]}"
        except:
            pass
        
        # Convert notes to ABC body
        abc_body = []
        measure_count = 0
        
        for element in score.flat.notesAndRests:
            if isinstance(element, music21.note.Note):
                abc_body.append(note_to_abc(element))
            elif isinstance(element, music21.note.Rest):
                dur = duration_to_abc(element.duration.quarterLength)
                abc_body.append("z" + dur)
            elif isinstance(element, music21.chord.Chord):
                if len(element.notes) > 0:
                    abc_body.append(note_to_abc(element.notes[0]))
            
            measure_count += 1
            if measure_count % 4 == 0:
                abc_body.append("|")
        
        body_str = "".join(abc_body)
        abc_lines.append(body_str)
        abc_str = "\n".join(abc_lines) if abc_lines else ""
        
        if abc_str:
            # Clean ABC string
            lines = abc_str.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('%'):
                    cleaned_lines.append(line)
            abc_str = '\n'.join(cleaned_lines)
            
            if len(abc_str.strip()) > 0:
                # Save ABC file
                abc_path = Path(output_dir) / "abc" / f"{Path(midi_file).stem}.abc"
                abc_path.parent.mkdir(parents=True, exist_ok=True)
                
                try:
                    with open(abc_path, 'w') as f:
                        f.write(abc_str)
                    
                    # Verify file was written
                    if abc_path.exists() and abc_path.stat().st_size > 0:
                        return (str(midi_file), abc_str)
                except Exception as e:
                    return None
        
        return None
    except Exception as e:
        return None
    finally:
        # Force garbage collection
        import gc
        gc.collect()

