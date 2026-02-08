import time
import random
import logging
from enum import Enum, auto

class DataStatus(Enum):
    VALID = auto()
    INVALID = auto()
    UNKNOWN = auto()

class Level(Enum):
    L0 = auto()
    L1 = auto()
    L2 = auto()
    L_NTC = auto()

class Mode(Enum):
    STAND_BY = auto()
    STAFF_RESPONSIBLE = auto()
    UNFITTED = auto()
    SYSTEM_NATIONAL = auto()
    SHUNTING = auto()
    NON_LEADING = auto()
    FULL_SUPERVISION = auto()
    ON_SIGHT = auto()
    LIMITED_SUPERVISION = auto()
    ENDED = auto()

class RadioNetwork(Enum):
    GSM_R = auto()
    FRMCS = auto()
    FRMCS_GSM_R = auto()

class MockController:
    """Provides a script of actions for the simulation to follow."""
    def __init__(self, actions: list):
        self.actions = actions
        self.action_index = 0
        self.history = []

    def next_action(self, prompt=""):
        """Returns the next scripted action."""
        if self.action_index >= len(self.actions):
            raise Exception(f"Test script ran out of actions. History: {self.history}")
        
        action = self.actions[self.action_index]
        self.action_index += 1
        self.history.append((prompt, action))
        return action

class ERTMSOnBoardSystem:
    def __init__(self, controller: MockController = None, logger: logging.Logger = None):
        self.controller = controller
        self.logger = logger if logger else logging.getLogger("start_of_mission")
        self.radio_systems_installed = [RadioNetwork.GSM_R, RadioNetwork.FRMCS]
        self.current_mode = Mode.STAND_BY
        self.desk_open = True
        self.communication_session_active = False
        self.driver_id_status = DataStatus.UNKNOWN
        self.level_status = DataStatus.UNKNOWN
        self.position_status = DataStatus.INVALID
        self.rbc_contact_status = DataStatus.UNKNOWN
        self.train_data_status = DataStatus.UNKNOWN
        self.train_running_number_status = DataStatus.UNKNOWN
        self.driver_id = None
        self.level = None
        self.position = "LRBG_4711_INVALID"
        self.rbc_contact = None
        self.train_data = None
        self.train_running_number = None
        self.radio_network_type = RadioNetwork.FRMCS_GSM_R
        self.gsmr_registered = False
        self.frmcs_registered = False

    def _simulate_driver_action(self, prompt, options=None):
        time.sleep(0.01) 
        if self.controller:
            return self.controller.next_action(f"Driver Action: {prompt}")

        if options:
            return random.choice(options)
        if "Driver ID" in prompt:
            return "DRIVER_007"
        if "Train Data" in prompt:
            return "Length: 200m, Cat: P"
        if "Train Running Number" in prompt:
            return "TRN-12345"
        return "SIMULATED_DATA"

    def _mock_radio_registration(self, radio_type):
        time.sleep(0.02)
        if self.controller:
            response = self.controller.next_action(f"Radio Registration: {radio_type.name}")
            self.gsmr_registered = response
            self.frmcs_registered = response
            return response
        
        self.gsmr_registered = True
        self.frmcs_registered = True
        return True

    def _mock_rbc(self, action, data=None):
        if self.controller:
            return self.controller.next_action(f"RBC Action: {action}")
        
        if action == "open_session":
            return "session_open" if random.choice([True, True, False]) else "session_failed"
        if action == "som_position_report":
            if data == "invalid":
                return random.choice(["validate_position", "cannot_validate"])
            return random.choice(["accept_train", "reject_train"])
        if action == "wait_for_train_data_ack":
            return "ack" if random.choice([True, True, False]) else "no_ack"
        if action == "request_ma":
            return random.choice(['ma_sr', 'ma_os', 'ma_fs'])
        return "unknown_response"
    

    def _procedure_s1_driver_id_entry(self):
        # BPMN: som_enterid_dmi_1 fires when driver_id_status is UNKNOWN (first entry)
        # BPMN: som_retry_dmi_1 fires when driver_id_status is INVALID (re-entry after failed validation)
        if self.driver_id_status == DataStatus.UNKNOWN:
            self.logger.info("som_enterid_dmi_1")
            self.driver_id = self._simulate_driver_action("Please enter Driver ID:")
        elif self.driver_id_status == DataStatus.INVALID:
            self.logger.info("som_retry_dmi_1")
            self.driver_id = self._simulate_driver_action("Please revalidate/re-enter Driver ID:")
        # BPMN: som_storeid_evc_1 — the EVC stores the entered/re-entered driver ID
        self.logger.info("som_storeid_evc_1")
        self.driver_id_status = DataStatus.VALID
        return 'D2'

    def _procedure_d2_check_pos_level(self):
        if self.position_status == DataStatus.VALID and self.level_status == DataStatus.VALID:
            return 'D3'
        else:
            return 'S2'
        
    def _procedure_d3_check_level_valid(self):
        if self.level == Level.L2:
            return 'D7'
        elif self.level in [Level.L0, Level.L1, Level.L_NTC]:
            return 'S10'
        
    def _procedure_s2_level_entry(self):
        level_input = None
        if self.level_status == DataStatus.UNKNOWN:
            level_input = self._simulate_driver_action("Please enter ERTMS/ETCS Level:", ['0', '1', '2', 'NTC'])
        elif self.level_status == DataStatus.INVALID:
            level_input = self._simulate_driver_action("Please re-validate/re-enter ERTMS/ETCS Level:", ['0', '1', '2', 'NTC'])

        if level_input == '0': self.level = Level.L0
        elif level_input == '1': self.level = Level.L1
        elif level_input == '2': self.level = Level.L2
        elif level_input == 'NTC': self.level = Level.L_NTC
        
        if self.level_status != DataStatus.VALID:
            self.level_status = DataStatus.VALID

        # BPMN: som_validate_evc_1 — the EVC validates the stored ID against the entered level
        self.logger.info("som_validate_evc_1")
        # BPMN: som_checklev_evc_1 — the EVC checks the validated level to route L2 vs L0/L1/NTC
        self.logger.info("som_checklev_evc_1")

        if self.level == Level.L2:
            return 'S3'
        elif self.level in [Level.L0, Level.L1, Level.L_NTC]:
            return 'S10'
        return 'S2'

    def _phase_1_get_initial_data(self):
        if not (self.current_mode == Mode.STAND_BY and self.desk_open):
            return None 

        state = self._procedure_s1_driver_id_entry()
        if state == 'D2':
            state = self._procedure_d2_check_pos_level()
        
        if state == 'S2':
            state = self._procedure_s2_level_entry()
        
        if state == 'D3':
            # Position and level were already valid; still need to validate and check level
            # BPMN: som_validate_evc_1
            self.logger.info("som_validate_evc_1")
            # BPMN: som_checklev_evc_1
            self.logger.info("som_checklev_evc_1")
            state = self._procedure_d3_check_level_valid()
            if state == 'D7': 
                return 'S3' 
        
        return state 

    def _procedure_d7_check_radio_registration(self):
        if self.radio_network_type == RadioNetwork.FRMCS_GSM_R and self.gsmr_registered and self.frmcs_registered:
            return 'A31'
        elif self.radio_network_type == RadioNetwork.GSM_R and self.gsmr_registered:
            return 'A31'
        elif self.radio_network_type == RadioNetwork.FRMCS and self.frmcs_registered:
            return 'A31'
        else:
            return 'S4'
        
    def _procedure_s4_wait_for_radio(self):
        success = self._mock_radio_registration(self.radio_network_type)
        if success:
            return 'A31'
        else:
            time.sleep(0.01)
            return 'A42'

    def _procedure_a42_d9_radio_fail(self):
        # BPMN: som_giveup_evc_1 — radio registration failed, EVC gives up on RBC connection
        self.logger.info("som_giveup_evc_1")
        return 'S10'

    def _procedure_a31_d31_a32_rbc_session(self):
        # BPMN: som_openconn_rtm_1 — RTM opens the communication session with RBC
        self.logger.info("som_openconn_rtm_1")
        response = self._mock_rbc("open_session")
        if response == "session_open":
            self.communication_session_active = True
            return 'D32'
        else:
            self.communication_session_active = False
            # BPMN: som_retryconn_dmi_1 — session failed, DMI prompts driver to retry
            self.logger.info("som_retryconn_dmi_1")
            return 'A32'

    def _procedure_d32_a33_a34_position_report(self):
        if self.position_status == DataStatus.VALID:
            # BPMN: som_storepos_rbc_1 — position is already valid, RBC stores it directly
            self.logger.info("som_storepos_rbc_1")
            # BPMN: som_storevalacc_rbc_1 — RBC stores the validation/acceptance result (converges with checktrain path)
            self.logger.info("som_storevalacc_rbc_1")
            # BPMN: som_storeacc_evc_1 — EVC stores the acceptance outcome locally
            self.logger.info("som_storeacc_evc_1")
            return 'S10'
        else:
            # BPMN: som_checkpos_rbc_1 — RBC checks the position report from OBS
            self.logger.info("som_checkpos_rbc_1")
            return 'D33'

    def _procedure_d33_acceptance(self):
        response = self._mock_rbc("som_position_report", 
                                   data="invalid" if self.position_status == DataStatus.INVALID else "no_position")

        if response == "validate_position":
            # BPMN: som_checktrain_rbc_1 — RBC validates the position, checks train acceptance
            self.logger.info("som_checktrain_rbc_1")
            self.position_status = DataStatus.VALID
        elif response == "accept_train":
            # BPMN: som_checktrain_rbc_1 — RBC accepted the train with unknown position
            self.logger.info("som_checktrain_rbc_1")
            self.position_status = DataStatus.UNKNOWN
        elif response == "reject_train" or response == "cannot_validate":
            # BPMN: som_checktrain_rbc_1 — RBC checked and rejected/cannot validate
            self.logger.info("som_checktrain_rbc_1")
            self.position_status = DataStatus.UNKNOWN
            self.communication_session_active = False 
        # BPMN: som_storevalacc_rbc_1 — RBC stores the validation/acceptance result
        self.logger.info("som_storevalacc_rbc_1")
        # BPMN: som_storeacc_evc_1 — EVC stores the acceptance outcome locally
        self.logger.info("som_storeacc_evc_1")
        return 'S10'

    def _phase_2_connect_to_rbc(self, start_state):
        current_state = start_state
        if current_state == 'S3':
            current_state = self._procedure_d7_check_radio_registration()

        if current_state == 'S4':
            current_state = self._procedure_s4_wait_for_radio()

        if current_state == 'A42':
            current_state = self._procedure_a42_d9_radio_fail()

        if current_state == 'A31':
            current_state = self._procedure_a31_d31_a32_rbc_session()

        if current_state == 'D32':
            current_state = self._procedure_d32_a33_a34_position_report()

        if current_state == 'D33':
            current_state = self._procedure_d33_acceptance()
        
        if current_state == 'A32':
            current_state = 'S10'
            
        return current_state

    def _procedure_s12_d12_s13_train_data(self):
        self.train_data = self._simulate_driver_action("Enter/re-validate Train Data:")
        self.train_data_status = DataStatus.VALID
        
        if self.train_running_number_status == DataStatus.VALID:
            return 'D10'
        
        self.train_running_number = self._simulate_driver_action("Enter/re-validate TRN:")
        self.train_running_number_status = DataStatus.VALID
        return 'D10'

    def _phase_3_get_train_data(self, start_state):
        options = ['TD', 'SH', 'NL']
        if self.level == Level.L2 and self.position_status == DataStatus.VALID: 
            options.append('SM')
        
        # BPMN: som_driversel_dmi_1 — DMI presents mode selection to driver
        self.logger.info("som_driversel_dmi_1")
        choice = self._simulate_driver_action(
            "Select: Train Data (TD), Shunting (SH), Non Leading (NL)" +
            (", Supervised Manoeuvre (SM)" if 'SM' in options else ""),
            options
        )
        
        if choice == 'TD':
            # BPMN: som_inserttraindata_dmi_1 — DMI collects train data from driver
            self.logger.info("som_inserttraindata_dmi_1")
            return self._procedure_s12_d12_s13_train_data()
        elif choice == 'SH':
            # BPMN: som_NLSHproc_evc_1 — EVC processes the NL/SH mode selection
            self.logger.info("som_NLSHproc_evc_1")
            self.current_mode = Mode.SHUNTING
            return None
        elif choice == 'NL':
            # BPMN: som_NLSHproc_evc_1 — EVC processes the NL/SH mode selection
            self.logger.info("som_NLSHproc_evc_1")
            self.current_mode = Mode.NON_LEADING
            return None
        elif choice == 'SM':
            # BPMN: som_NLSHproc_evc_1 — EVC processes the SM selection (same convergence point)
            self.logger.info("som_NLSHproc_evc_1")
            self.current_mode = Mode.FULL_SUPERVISION 
            return None
        
        return 'S10'

    def _procedure_d10_d11_s11_data_ack(self):
        if self.level != Level.L2:
            return 'S20'

        if not self.communication_session_active:
            return 'S10'
        
        # BPMN: som_checkrbcsess_rtm_1 — RTM checks the RBC session is active before sending
        self.logger.info("som_checkrbcsess_rtm_1")
        max_retries = 10 
        for _ in range(max_retries):
            if self._mock_rbc("wait_for_train_data_ack") == "ack":
                return 'S20'
            else:
                time.sleep(0.01)
        
        return 'S10'

    def _procedure_s22_s23_s24_mode_grant(self, next_state):
        if next_state == 'S22':
            # BPMN: som_grantSN_evc_1 — EVC grants System National mode
            self.logger.info("som_grantSN_evc_1")
            self.logger.info("som_awaitack_dmi_1")
            self._simulate_driver_action("Acknowledge running under National System supervision.")
            self.current_mode = Mode.SYSTEM_NATIONAL
        elif next_state == 'S23':
            # BPMN: som_grantUN_evc_1 — EVC grants Unfitted mode
            self.logger.info("som_grantUN_evc_1")
            self.logger.info("som_awaitack_dmi_1")
            self._simulate_driver_action("Acknowledge running in Unfitted mode.")
            self.current_mode = Mode.UNFITTED
        elif next_state == 'S24':
            # BPMN: som_grantSR_evc_1 — EVC grants Staff Responsible mode
            self.logger.info("som_grantSR_evc_1")
            self.logger.info("som_awaitack_dmi_1")
            self._simulate_driver_action("Acknowledge running in Staff Responsible mode.")
            self.current_mode = Mode.STAFF_RESPONSIBLE
        
        # BPMN: som_chmod_evc_1 — EVC finalises the mode change
        self.logger.info("som_chmod_evc_1")
        if self.position_status == DataStatus.INVALID:
            self.position_status = DataStatus.UNKNOWN
        return None

    def _procedure_s21_s25_l2_grant(self):
        # BPMN: som_selstart_dmi_1 — already logged before this is called; here we send the MA request
        # BPMN: som_sendMAreq_rtm_1 — RTM sends the Movement Authority request to RBC
        self.logger.info("som_sendMAreq_rtm_1")
        response = self._mock_rbc("request_ma")
        
        # BPMN: som_checktrainroute_rbc_1 — RBC checks the train route feasibility
        self.logger.info("som_checktrainroute_rbc_1")
        # BPMN: som_checkval_rbc_1 — RBC validates the MA grant conditions
        self.logger.info("som_checkval_rbc_1")

        if response == 'ma_sr':
            # BPMN: som_grantSR_rbc_1 — RBC grants Staff Responsible
            self.logger.info("som_grantSR_rbc_1")
            return 'S24'
        elif response == 'ma_os':
            # BPMN: som_grantOS_rbc_1 — RBC grants On Sight
            self.logger.info("som_grantOS_rbc_1")
            self.logger.info("som_awaitack_dmi_1")
            self._simulate_driver_action("Acknowledge running in OS/LS/SH mode.")
            self.current_mode = random.choice([Mode.ON_SIGHT, Mode.LIMITED_SUPERVISION, Mode.SHUNTING])
            # BPMN: som_chmod_evc_1
            self.logger.info("som_chmod_evc_1")
            return None
        elif response == 'ma_fs':
            # BPMN: som_grantFS_rbc_1 — RBC grants Full Supervision
            self.logger.info("som_grantFS_rbc_1")
            self.logger.info("som_awaitack_dmi_1")
            self.current_mode = Mode.FULL_SUPERVISION
            # BPMN: som_chmod_evc_1
            self.logger.info("som_chmod_evc_1")
            return None
        return 'S10'

    def _phase_4_assign_supervision(self, start_state):
        current_state = start_state
        
        if current_state == 'D10':
            current_state = self._procedure_d10_d11_s11_data_ack()
            if current_state == 'S10':
                return 'S10'

        if current_state == 'S20':
            # BPMN: som_selstart_dmi_1 — DMI presents "Start" to the driver
            self.logger.info("som_selstart_dmi_1")
            self._simulate_driver_action("Select 'Start'")
            if self.level == Level.L_NTC: current_state = 'S22'
            elif self.level == Level.L0: current_state = 'S23'
            elif self.level == Level.L1: current_state = 'S24'
            elif self.level == Level.L2: current_state = 'S21'
            else: current_state = 'S10'

        if current_state in ['S22', 'S23', 'S24']:
            current_state = self._procedure_s22_s23_s24_mode_grant(current_state)
        elif current_state == 'S21':
            current_state = self._procedure_s21_s25_l2_grant()
            if current_state == 'S24': 
                current_state = self._procedure_s22_s23_s24_mode_grant(current_state)

        return current_state

    def run_start_of_mission(self):
        
        current_state = self._phase_1_get_initial_data()
        if current_state is None:
            return 

        if current_state == 'S3':
            current_state = self._phase_2_connect_to_rbc(current_state)
        
        while current_state == 'S10':
            current_state = self._phase_3_get_train_data(current_state)
            if current_state is None:
                # BPMN: som_awaitack_dmi_1 and som_chmod_evc_1 for NL/SH/SM terminal paths
                self.logger.info("som_awaitack_dmi_1")
                self.logger.info("som_chmod_evc_1")
                return 

            if current_state == 'D10':
                current_state = self._phase_4_assign_supervision(current_state)
                if current_state is None:
                    return 
                
                if current_state == 'S10':
                    continue