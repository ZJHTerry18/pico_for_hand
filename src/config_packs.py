default_loss_weights = {
    'lw_contact': 8.0,
    'lw_silhouette': 0.3,
    'lw_silhouette_distance': 0.3,
    'lw_scale': 0,
    'lw_collision_p2': 1.0,
    'lw_collision_p3': 50,
    'lw_pose_reg': 0.05,
    'lw_silhouette_hand': 0.1,
}

class ConfigPack:
    def __init__(
        self,
        # input file names
        hand_inference_file: str = 'hand_posed_mesh.ply',
        # hand_detection_file: str = 'hand_detection.npz',
        object_mesh_file: str = 'object.obj',
        # object_detection_file: str = 'object_detection.npz',
        contact_mapping_file: str = 'corresponding_contacts.json',
        # support file names
        sparse_dense_mapping_file: str = 'sparse_dense_mapping.json',
        # optimization nr of steps
        nr_phase_1_steps: int = 250,
        lr_rotation_phase_1: float = 0.04,
        lr_translation_phase_1: float = 0.02,
        skip_phase_1: bool = False,
        nr_phase_2_steps: int = 1500,
        lr_rotation_phase_2: float = 0.04,
        lr_translation_phase_2: float = 0.03,
        lr_scaling_phase_2: float = 0.02,
        skip_phase_2: bool = False,
        nr_phase_3_steps: int = 1000,
        skip_phase_3: bool = False,
        loss_weights: dict = default_loss_weights,
    ):
        self.hand_inference_file = hand_inference_file
        # self.hand_detection_file = hand_detection_file
        self.object_mesh_file = object_mesh_file
        # self.object_detection_file = object_detection_file
        self.contact_mapping_file = contact_mapping_file
        
        self.sparse_dense_mapping_file = sparse_dense_mapping_file

        self.nr_phase_1_steps = nr_phase_1_steps
        self.lr_rotation_phase_1 = lr_rotation_phase_1
        self.lr_translation_phase_1 = lr_translation_phase_1
        self.skip_phase_1 = skip_phase_1
        self.nr_phase_2_steps = nr_phase_2_steps
        self.lr_rotation_phase_2 = lr_rotation_phase_2
        self.lr_translation_phase_2 = lr_translation_phase_2
        self.lr_scaling_phase_2 = lr_scaling_phase_2
        self.skip_phase_2 = skip_phase_2
        self.nr_phase_3_steps = nr_phase_3_steps
        self.skip_phase_3 = skip_phase_3

        self.loss_weights = loss_weights

default_config = ConfigPack()

arctic_config = ConfigPack(
    nr_phase_1_steps=5000,
    lr_rotation_phase_1=0.2,
    lr_translation_phase_1=0.1,
    skip_phase_1=False,
    nr_phase_2_steps=500,
    lr_rotation_phase_2=0.01,
    lr_translation_phase_2=0.005,
    lr_scaling_phase_2=0.005,
    skip_phase_2=False,
    nr_phase_3_steps=1000,
    skip_phase_3=True,
    loss_weights=default_loss_weights,
)

CONFIGS_FACTORY = {
    "default": default_config,
    "arctic": arctic_config,
}