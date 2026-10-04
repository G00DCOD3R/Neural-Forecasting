import h5py
import numpy as np

class MicronsReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.f = h5py.File(self.file_path, 'r')

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _encode_hash(self, h):
        return h.replace('/', '%2F')

    def _decode_hash(self, h):
        return h.replace('%2F', '/')

    def _read_trial(self, trial_grp, session_key, brain_area=None):
        """
        Helper to extract all datasets from a trial group.
        Centralizes field names so they only need updating in one place.
        """
        if brain_area:
            area_path = f"sessions/{session_key}/meta/area_indices/{brain_area}"
            if area_path not in self.f:
                return None
            indices = self.f[area_path][:]
            responses = trial_grp['responses'][indices, :]
        else:
            responses = trial_grp['responses'][:]

        return {
            'session':    session_key,
            'trial_idx':  trial_grp.name.split('/')[-1],
            'responses':  responses,
            'treadmill':  trial_grp['treadmill'][:],
            'pupil':      trial_grp['pupil'][:],
            'stim_times': trial_grp['stim_times'][:],
        }

    def get_full_data_by_hash(self, condition_hash, brain_area=None):
        """
        Returns a dictionary with the clip and all trials (responses, treadmill,
        pupil, stim_times) associated with a condition hash.

        Args:
            condition_hash (str): The identifier for the video.
            brain_area (str, optional): Filter for neural responses.

        Returns:
            dict: {
                'clip': np.array,
                'stim_type': str,
                'trials': [
                    {'session': str, 'trial_idx': str, 'responses': np.array,
                     'treadmill': np.array, 'pupil': np.array, 'stim_times': np.array},
                    ...
                ]
            }
        """
        h_key = self._encode_hash(condition_hash)
        clip, stim_type = self.get_video_data(condition_hash)
        if clip is None:
            return None

        data_out = {'clip': clip, 'stim_type': stim_type, 'trials': []}

        instances = self.f[f'videos/{h_key}/instances']
        for instance_name in instances:
            trial_grp = instances[instance_name]
            session_key = "_".join(instance_name.split('_')[:2])
            trial = self._read_trial(trial_grp, session_key, brain_area)
            if trial is not None:
                data_out['trials'].append(trial)

        return data_out

    def get_responses_by_hash(self, condition_hash, brain_area=None):
        """Retrieves only neural responses associated with a hash across sessions."""
        full_data = self.get_full_data_by_hash(condition_hash, brain_area=brain_area)
        if full_data is None:
            return []
        return [
            {
                'session':   t['session'],
                'trial_idx': t['trial_idx'],
                'responses': t['responses'],
            }
            for t in full_data['trials']
        ]

    def get_video_data(self, condition_hash):
        h_key = self._encode_hash(condition_hash)
        video_path = f"videos/{h_key}"
        if video_path not in self.f:
            return None, None
        vid_grp = self.f[video_path]
        clip = vid_grp['clip'][:]
        stim_type = vid_grp.attrs.get('type', 'Unknown')
        return clip, stim_type

    def get_hashes_by_session(self, session_key, return_unique=False):
        """Returns condition hashes shown in a specific session."""
        if session_key not in self.f['sessions']:
            raise ValueError(f"Session {session_key} not found.")
        hashes = self.f[f'sessions/{session_key}/meta/condition_hashes'][:]
        decoded = [self._decode_hash(h.decode('utf-8')) for h in hashes]
        return set(decoded) if return_unique else decoded

    def get_hashes_by_type(self, stim_type):
        """Returns hashes belonging to a specific stimulus type (e.g., 'Monet2')."""
        if stim_type not in self.f['types']:
            return []
        return [self._decode_hash(k) for k in self.f[f'types/{stim_type}'].keys()]

    def get_available_brain_areas(self, session_key=None):
        """Returns brain areas available in the file or a specific session."""
        if session_key:
            return list(self.f[f'sessions/{session_key}/meta/area_indices'].keys())
        return list(self.f['brain_areas'].keys())

    def get_trial(self, session_key, trial_idx, brain_area=None):
        """
        Direct access to a single trial by session and trial index.

        Args:
            session_key (str): e.g. '4_1'
            trial_idx (int or str): trial index
            brain_area (str, optional): filter responses by area
        """
        trial_path = f"sessions/{session_key}/trials/{trial_idx}"
        if trial_path not in self.f:
            raise ValueError(f"Trial {trial_idx} not found in session {session_key}.")
        return self._read_trial(self.f[trial_path], session_key, brain_area)

    def print_structure(self, max_items=5, follow_links=False):
        """Prints a tree-like representation of the HDF5 database."""
        print(f"\nStructure of: {self.file_path}")
        print("=" * 50)

        def _print_tree(name, obj, indent="", current_key=""):
            item_name = current_key if current_key else name
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}📄 {item_name:20} [Dataset: {obj.shape}, {obj.dtype}]")
                return
            attrs = dict(obj.attrs)
            attr_str = f"  | Attributes: {attrs}" if attrs else ""
            print(f"{indent}📂 {item_name.upper()}/ {attr_str}")
            keys = sorted(obj.keys())
            for key in keys[:max_items]:
                link_obj = obj.get(key, getlink=True)
                if isinstance(link_obj, h5py.SoftLink):
                    if follow_links:
                        _print_tree(key, obj[key], indent + "    ", current_key=key)
                    else:
                        print(f"{indent}    🔗 {key:18} -> {link_obj.path}")
                else:
                    _print_tree(key, obj[key], indent + "    ", current_key=key)
            if len(keys) > max_items:
                print(f"{indent}    ... and {len(keys) - max_items} more items")

        for key in sorted(self.f.keys()):
            _print_tree(key, self.f[key], current_key=key)