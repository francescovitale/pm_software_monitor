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
        self.logger = logger
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

    def _log(self, activity_name):
        if self.logger:
            self.logger.info(activity_name)

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
        # BPMN: som_enterid_dmi_1 is the DMI prompt for driver ID.
        # som_retry_dmi_1 is logged when the status is already INVALID (a retry).
        if self.driver_id_status == DataStatus.INVALID:
            self._log("som_retry_dmi_1")
        self._log("som_enterid_dmi_1")
        if self.driver_id_status == DataStatus.UNKNOWN:
            self.driver_id = self._simulate_driver_action("Please enter Driver ID:")
        elif self.driver_id_status == DataStatus.INVALID:
            self.driver_id = self._simulate_driver_action("Please revalidate/re-enter Driver ID:")
        # BPMN: after entry, storeid_evc_1 stores the ID, validate_evc_1 validates it.
        # The original code unconditionally sets status to VALID (no retry loop in this
        # prototype), so storeid and validate always execute.
        self._log("som_storeid_evc_1")
        self.driver_id_status = DataStatus.VALID
        self._log("som_validate_evc_1")
        return 'D2'

    def _procedure_d2_check_pos_level(self):
        if self.position_status == DataStatus.VALID and self.level_status == DataStatus.VALID:
            return 'D3'
        else:
            return 'S2'
        
    def _procedure_d3_check_level_valid(self):
        # BPMN: som_checklev_evc_1 evaluates the level.
        self._log("som_checklev_evc_1")
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

        # BPMN: som_checklev_evc_1 evaluates the level after entry.
        self._log("som_checklev_evc_1")
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
        # BPMN: som_giveup_evc_1 fires when radio registration fails completely.
        self._log("som_giveup_evc_1")
        return 'S10'

    def _procedure_a31_d31_a32_rbc_session(self):
        # BPMN: som_openconn_rtm_1 is the attempt to open an RBC session.
        self._log("som_openconn_rtm_1")
        response = self._mock_rbc("open_session")
        if response == "session_open":
            self.communication_session_active = True
            return 'D32'
        else:
            self.communication_session_active = False
            # BPMN: som_retryconn_dmi_1 fires on session failure (retry prompt).
            # The prototype does not loop back; it falls through to A32 -> S10.
            self._log("som_retryconn_dmi_1")
            return 'A32'

    def _procedure_d32_a33_a34_position_report(self):
        # BPMN: som_checkpos_rbc_1 checks position validity at the RBC.
        self._log("som_checkpos_rbc_1")
        if self.position_status == DataStatus.VALID:
            # BPMN: valid-pos branch -> som_storepos_rbc_1
            self._log("som_storepos_rbc_1")
            # BPMN: both branches converge at som_storevalacc_rbc_1
            self._log("som_storevalacc_rbc_1")
            # BPMN: som_storeacc_evc_1
            self._log("som_storeacc_evc_1")
            return 'S10'
        else:
            return 'D33'

    def _procedure_d33_acceptance(self):
        response = self._mock_rbc("som_position_report", 
                                   data="invalid" if self.position_status == DataStatus.INVALID else "no_position")

        if response == "validate_position":
            self.position_status = DataStatus.VALID
            # BPMN: invalid-pos branch -> som_checktrain_rbc_1 (RBC validates)
            self._log("som_checktrain_rbc_1")
        elif response == "accept_train":
            self.position_status = DataStatus.UNKNOWN
            # BPMN: invalid-pos branch -> som_checktrain_rbc_1
            self._log("som_checktrain_rbc_1")
        elif response == "reject_train" or response == "cannot_validate":
            self.position_status = DataStatus.UNKNOWN
            self.communication_session_active = False
            # BPMN: invalid-pos branch -> som_checktrain_rbc_1
            self._log("som_checktrain_rbc_1")

        # BPMN: both branches converge at som_storevalacc_rbc_1
        self._log("som_storevalacc_rbc_1")
        # BPMN: som_storeacc_evc_1
        self._log("som_storeacc_evc_1")
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
        # BPMN: som_inserttraindata_dmi_1 prompts for train data / TRN.
        self._log("som_inserttraindata_dmi_1")
        self.train_data = self._simulate_driver_action("Enter/re-validate Train Data:")
        self.train_data_status = DataStatus.VALID
        
        if self.train_running_number_status == DataStatus.VALID:
            return 'D10'
        
        self.train_running_number = self._simulate_driver_action("Enter/re-validate TRN:")
        self.train_running_number_status = DataStatus.VALID
        return 'D10'

    def _phase_3_get_train_data(self, start_state):
        # BPMN: som_driversel_dmi_1 presents the selection menu to the driver.
        self._log("som_driversel_dmi_1")
        options = ['TD', 'SH', 'NL']
        if self.level == Level.L2 and self.position_status == DataStatus.VALID: 
            options.append('SM')
        
        choice = self._simulate_driver_action(
            "Select: Train Data (TD), Shunting (SH), Non Leading (NL)" +
            (", Supervised Manoeuvre (SM)" if 'SM' in options else ""),
            options
        )
        
        if choice == 'TD':
            return self._procedure_s12_d12_s13_train_data()
        elif choice == 'SH':
            # BPMN: som_NLSHproc_evc_1 handles NL/SH mode transitions.
            self._log("som_NLSHproc_evc_1")
            self.current_mode = Mode.SHUNTING
            # BPMN: som_chmod_evc_1 finalises the mode change at end.
            self._log("som_chmod_evc_1")
            return None
        elif choice == 'NL':
            # BPMN: som_NLSHproc_evc_1
            self._log("som_NLSHproc_evc_1")
            self.current_mode = Mode.NON_LEADING
            # BPMN: som_chmod_evc_1
            self._log("som_chmod_evc_1")
            return None
        elif choice == 'SM':
            self.current_mode = Mode.FULL_SUPERVISION 
            # BPMN: som_chmod_evc_1
            self._log("som_chmod_evc_1")
            return None
        
        return 'S10'

    def _procedure_d10_d11_s11_data_ack(self):
        # BPMN: som_checkrbcsess_rtm_1 verifies the RBC session is still active.
        self._log("som_checkrbcsess_rtm_1")
        if self.level != Level.L2:
            return 'S20'

        if not self.communication_session_active:
            return 'S10'
        
        max_retries = 10 
        for _ in range(max_retries):
            if self._mock_rbc("wait_for_train_data_ack") == "ack":
                return 'S20'
            else:
                time.sleep(0.01)
        
        return 'S10'

    def _procedure_s22_s23_s24_mode_grant(self, next_state):
        if next_state == 'S22':
            # BPMN: som_grantSN_evc_1 for National System
            self._log("som_grantSN_evc_1")
            self._simulate_driver_action("Acknowledge running under National System supervision.")
            self.current_mode = Mode.SYSTEM_NATIONAL
        elif next_state == 'S23':
            # BPMN: som_grantUN_evc_1 for Unfitted
            self._log("som_grantUN_evc_1")
            self._simulate_driver_action("Acknowledge running in Unfitted mode.")
            self.current_mode = Mode.UNFITTED
        elif next_state == 'S24':
            # BPMN: som_grantSR_evc_1 for Staff Responsible
            self._log("som_grantSR_evc_1")
            self._simulate_driver_action("Acknowledge running in Staff Responsible mode.")
            self.current_mode = Mode.STAFF_RESPONSIBLE
        
        if self.position_status == DataStatus.INVALID:
            self.position_status = DataStatus.UNKNOWN
        # BPMN: som_awaitack_dmi_1 awaits driver acknowledgement.
        self._log("som_awaitack_dmi_1")
        # BPMN: som_chmod_evc_1 finalises the mode.
        self._log("som_chmod_evc_1")
        return None

    def _procedure_s21_s25_l2_grant(self):
        # BPMN: som_selstart_dmi_1 is the "Start" selection for L2.
        self._log("som_selstart_dmi_1")
        # BPMN: som_sendMAreq_rtm_1 sends the MA request to the RBC.
        self._log("som_sendMAreq_rtm_1")
        response = self._mock_rbc("request_ma")
        # BPMN: som_checktrainroute_rbc_1 — RBC checks the train route.
        self._log("som_checktrainroute_rbc_1")
        # BPMN: som_checkval_rbc_1 — RBC validates and decides the grant type.
        self._log("som_checkval_rbc_1")
        
        if response == 'ma_sr':
            # BPMN: som_grantSR_rbc_1 — RBC grants Staff Responsible.
            self._log("som_grantSR_rbc_1")
            return 'S24'
        elif response == 'ma_os':
            # BPMN: som_grantOS_rbc_1 — RBC grants On Sight / LS / SH.
            self._log("som_grantOS_rbc_1")
            self._simulate_driver_action("Acknowledge running in OS/LS/SH mode.")
            self.current_mode = random.choice([Mode.ON_SIGHT, Mode.LIMITED_SUPERVISION, Mode.SHUNTING])
            # BPMN: som_awaitack_dmi_1
            self._log("som_awaitack_dmi_1")
            # BPMN: som_chmod_evc_1
            self._log("som_chmod_evc_1")
            return None
        elif response == 'ma_fs':
            # BPMN: som_grantFS_rbc_1 — RBC grants Full Supervision.
            self._log("som_grantFS_rbc_1")
            self.current_mode = Mode.FULL_SUPERVISION
            # BPMN: som_awaitack_dmi_1
            self._log("som_awaitack_dmi_1")
            # BPMN: som_chmod_evc_1
            self._log("som_chmod_evc_1")
            return None
        return 'S10'

    def _phase_4_assign_supervision(self, start_state):
        current_state = start_state
        
        if current_state == 'D10':
            current_state = self._procedure_d10_d11_s11_data_ack()
            if current_state == 'S10':
                return 'S10'

        if current_state == 'S20':
            # BPMN: som_selstart_dmi_1 for non-L2 paths (Start selection).
            if self.level != Level.L2:
                self._log("som_selstart_dmi_1")
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
                return 

            if current_state == 'D10':
                current_state = self._phase_4_assign_supervision(current_state)
                if current_state is None:
                    return 
                
                if current_state == 'S10':
                    continue