import pytest
import logging
import os
import shutil
from start_of_mission import (
    ERTMSOnBoardSystem, 
    MockController, 
    Mode, 
    Level, 
    DataStatus
)

# --- THIS FIXTURE IS NOW CORRECTED ---

@pytest.fixture(scope="session", autouse=True)
def clean_log_directory():
    """Cleans the log directory once before all tests."""
    log_dir = "test_logs"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

@pytest.fixture
def setup_test_logger(request, clean_log_directory):
    """
    This fixture runs for every test.
    It creates a unique logger for the test and configures it
    to write to its own file in the 'test_logs' directory.
    """
    log_dir = "test_logs"
    test_name = request.node.name
    log_file_path = os.path.join(log_dir, f"{test_name}.log")
    
    # 2. Create a logger
    logger = logging.getLogger(test_name)
    logger.setLevel(logging.DEBUG)
    
    # --- FIX ---
    # 1. Stop messages from going to pytest's root logger
    logger.propagate = False
    
    # 2. Remove any handlers added by pytest or previous runs
    if logger.hasHandlers():
        logger.handlers.clear()
    # --- END FIX ---

    # 3. Create a file handler
    handler = logging.FileHandler(log_file_path, mode='w')
    handler.setLevel(logging.DEBUG)
    
    # 4. Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # 5. Add the handler to the logger (this will now always run)
    logger.addHandler(handler)
    
    # 6. Yield the logger to the test
    yield logger
    
    # 7. Teardown
    for h in logger.handlers[:]:
        h.flush()
        h.close()
        logger.removeHandler(h)

# --- Category 1: Guard Clause & Basic Selection Tests (5 Tests) ---

def test_path_guard_desk_closed(setup_test_logger):
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=None)
    sim.desk_open = False
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.STAND_BY

def test_path_guard_not_in_stand_by(setup_test_logger):
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=None)
    sim.current_mode = Mode.FULL_SUPERVISION
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.FULL_SUPERVISION

def test_path_l1_select_shunting_immediately(setup_test_logger):
    script = ["DRIVER_007", '1', 'SH']
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.SHUNTING

def test_path_l0_select_non_leading_immediately(setup_test_logger):
    script = ["DRIVER_007", '0', 'NL']
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.NON_LEADING

def test_path_l2_select_supervised_manoeuvre_immediately(setup_test_logger):
    script = [
        "DRIVER_007", '2', True, "session_open", "validate_position", 'SM'
    ]
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.FULL_SUPERVISION


# --- Category 2: L2 Happy Paths (5 Tests) ---

def test_path_l2_success_grant_fs(setup_test_logger):
    script = [
        "DRIVER_007", '2', True, "session_open", "validate_position",
        'TD', "DATA", "TRN", "ack", "Start", "ma_fs"
    ]
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.FULL_SUPERVISION

def test_path_l2_success_grant_os(setup_test_logger):
    script = [
        "DRIVER_007", '2', True, "session_open", "validate_position",
        'TD', "DATA", "TRN", "ack", "Start", "ma_os", "Ack OS/LS/SH"
    ]
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode in [Mode.ON_SIGHT, Mode.LIMITED_SUPERVISION, Mode.SHUNTING]

def test_path_l2_success_grant_sr(setup_test_logger):
    script = [
        "DRIVER_007", '2', True, "session_open", "validate_position",
        'TD', "DATA", "TRN", "ack", "Start", "ma_sr", "Ack SR"
    ]
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.STAFF_RESPONSIBLE

def test_path_l2_success_position_already_valid(setup_test_logger):
    script = [
        "DRIVER_007", '2', True, "session_open", 
        'TD', "DATA", "TRN", "ack", "Start", "ma_fs"
    ]
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.position_status = DataStatus.VALID
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.FULL_SUPERVISION

def test_path_l2_success_trn_already_valid(setup_test_logger):
    script = [
        "DRIVER_007", '2', True, "session_open", "validate_position",
        'TD', "DATA",
        "ack", "Start", "ma_fs"
    ]
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.train_running_number_status = DataStatus.VALID
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.FULL_SUPERVISION


# --- Category 3: L0, L1, NTC Happy Paths (5 Tests) ---

def test_path_l1_success_grant_sr(setup_test_logger):
    script = ["DRIVER_007", '1', 'TD', "DATA", "TRN", "Start", "Ack SR"]
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.STAFF_RESPONSIBLE

def test_path_l0_success_grant_un(setup_test_logger):
    script = ["DRIVER_007", '0', 'TD', "DATA", "TRN", "Start", "Ack UN"]
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.UNFITTED

def test_path_ntc_success_grant_sn(setup_test_logger):
    script = ["DRIVER_007", 'NTC', 'TD', "DATA", "TRN", "Start", "Ack SN"]
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.SYSTEM_NATIONAL

def test_path_l1_success_trn_already_valid(setup_test_logger):
    script = ["DRIVER_007", '1', 'TD', "DATA", "Start", "Ack SR"]
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.train_running_number_status = DataStatus.VALID
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.STAFF_RESPONSIBLE

def test_path_l0_success_td_and_trn_already_valid(setup_test_logger):
    script = [
        "DRIVER_007", '0', 'TD', 
        "PREFILLED_DATA",
        "Start", "Ack UN"
    ]
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.train_data_status = DataStatus.VALID
    sim.train_running_number_status = DataStatus.VALID
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.UNFITTED


# --- Category 4: L2 Radio Failure Paths (5 Tests) ---

def test_path_l2_radio_fail_fallback_sh(setup_test_logger):
    script = ["DRIVER_007", '2', False, 'SH']
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.SHUNTING

def test_path_l2_radio_fail_fallback_nl(setup_test_logger):
    script = ["DRIVER_007", '2', False, 'NL']
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.NON_LEADING

def test_path_l2_radio_fail_loop_td_then_sh(setup_test_logger):
    script = ["DRIVER_007", '2', False, 'TD', "DATA", "TRN", 'SH']
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.SHUNTING

def test_path_l2_radio_fail_loop_td_then_nl(setup_test_logger):
    script = ["DRIVER_007", '2', False, 'TD', "DATA", "TRN", 'NL']
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.NON_LEADING

def test_path_l2_radio_fail_loop_td_twice_then_sh(setup_test_logger):
    script = [
        "DRIVER_007", '2', False,
        'TD', "DATA_1", "TRN_1",
        'TD', "DATA_2", "TRN_2",
        'SH'
    ]
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.SHUNTING


# --- Category 5: L2 RBC Session Failure Paths (5 Tests) ---

def test_path_l2_session_fail_fallback_sh(setup_test_logger):
    script = ["DRIVER_007", '2', True, "session_failed", 'SH']
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.SHUNTING

def test_path_l2_session_fail_fallback_nl(setup_test_logger):
    script = ["DRIVER_007", '2', True, "session_failed", 'NL']
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.NON_LEADING

def test_path_l2_session_fail_loop_td_then_sh(setup_test_logger):
    script = ["DRIVER_007", '2', True, "session_failed", 'TD', "DATA", "TRN", 'SH']
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.SHUNTING

def test_path_l2_session_fail_loop_td_then_nl(setup_test_logger):
    script = ["DRIVER_007", '2', True, "session_failed", 'TD', "DATA", "TRN", 'NL']
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.NON_LEADING

def test_path_l2_session_fail_loop_td_twice_then_sh(setup_test_logger):
    script = [
        "DRIVER_007", '2', True, "session_failed",
        'TD', "DATA_1", "TRN_1",
        'TD', "DATA_2", "TRN_2",
        'SH'
    ]
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.SHUNTING


# --- Category 6: L2 Position Report Failure Paths (5 Tests) ---

def test_path_l2_pos_reject_fallback_sh(setup_test_logger):
    script = ["DRIVER_007", '2', True, "session_open", "reject_train", 'SH']
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.SHUNTING

def test_path_l2_pos_reject_fallback_nl(setup_test_logger):
    script = ["DRIVER_007", '2', True, "session_open", "reject_train", 'NL']
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.NON_LEADING

def test_path_l2_pos_reject_loop_td_then_sh(setup_test_logger):
    script = [
        "DRIVER_007", '2', True, "session_open", "reject_train",
        'TD', "DATA", "TRN", 'SH'
    ]
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.SHUNTING

def test_path_l2_pos_reject_loop_td_then_nl(setup_test_logger):
    script = [
        "DRIVER_007", '2', True, "session_open", "reject_train",
        'TD', "DATA", "TRN", 'NL'
    ]
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.NON_LEADING

def test_path_l2_pos_accept_train_unknown_pos(setup_test_logger):
    script = [
        "DRIVER_007", '2', True, "session_open", "accept_train",
        'TD', "DATA", "TRN", "ack", "Start", "ma_fs"
    ]
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.FULL_SUPERVISION
    assert sim.position_status == DataStatus.UNKNOWN


# --- Category 7: L2 ACK Failure Paths (5 Tests) ---

def test_path_l2_ack_fail_fallback_sh(setup_test_logger):
    script = [
        "DRIVER_007", '2', True, "session_open", "validate_position",
        'TD', "DATA", "TRN",
    ]
    script.extend(["no_ack"] * 10)
    script.append('SH')
    
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.SHUNTING

def test_path_l2_ack_fail_fallback_nl(setup_test_logger):
    script = [
        "DRIVER_007", '2', True, "session_open", "validate_position",
        'TD', "DATA", "TRN",
    ]
    script.extend(["no_ack"] * 10)
    script.append('NL')
    
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.NON_LEADING

def test_path_l2_ack_fail_loop_td_then_sh(setup_test_logger):
    script = [
        "DRIVER_007", '2', True, "session_open", "validate_position",
        'TD', "DATA_1", "TRN_1",
    ]
    script.extend(["no_ack"] * 10)
    script.extend(['TD', "DATA_2", "TRN_2"])
    script.extend(["no_ack"] * 10)
    script.append('SH')
    
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.SHUNTING

def test_path_l2_ack_fail_loop_td_then_nl(setup_test_logger):
    script = [
        "DRIVER_007", '2', True, "session_open", "validate_position",
        'TD', "DATA_1", "TRN_1",
    ]
    script.extend(["no_ack"] * 10)
    script.extend(['TD', "DATA_2", "TRN_2"])
    script.extend(["no_ack"] * 10)
    script.append('NL')
    
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.NON_LEADING

def test_path_l2_ack_fail_max_retries_fallback_sh(setup_test_logger):
    script = [
        "DRIVER_007", '2', True, "session_open", "validate_position",
        'TD', "DATA", "TRN",
    ]
    script.extend(["no_ack"] * 10)
    script.append('SH')

    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.SHUNTING


# --- Category 8: Data Pre-Condition Paths (5 Tests) ---

def test_path_l2_data_prefilled_trn_needed(setup_test_logger):
    script = [
        "DRIVER_007", '2', True, "session_open", "validate_position",
        'TD', "PREFILLED_DATA", "TRN",
        "ack", "Start", "ma_fs"
    ]
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.train_data_status = DataStatus.VALID
    sim.train_running_number_status = DataStatus.UNKNOWN 
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.FULL_SUPERVISION

def test_path_l2_success_all_data_prefilled(setup_test_logger):
    script = [
        "DRIVER_007", '2', True, "session_open", "validate_position",
        'TD', "PREFILLED_DATA",
        "ack", "Start", "ma_fs"
    ]
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.train_data_status = DataStatus.VALID
    sim.train_running_number_status = DataStatus.VALID
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.FULL_SUPERVISION

def test_path_l1_success_all_data_prefilled(setup_test_logger):
    script = [
        "DRIVER_007", '1', 'TD', "PREFILLED_DATA", "Start", "Ack SR"
    ]
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.train_data_status = DataStatus.VALID
    sim.train_running_number_status = DataStatus.VALID
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.STAFF_RESPONSIBLE

def test_path_l2_driver_id_prefilled(setup_test_logger):
    script = [
        '2', True, "session_open", "validate_position",
        'TD', "DATA", "TRN", "ack", "Start", "ma_fs"
    ]
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.driver_id_status = DataStatus.VALID
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.FULL_SUPERVISION

def test_path_l2_level_prefilled(setup_test_logger):
    script = [
        "DRIVER_007",
        True, "session_open", "validate_position",
        'TD', "DATA", "TRN", "ack", "Start", "ma_fs"
    ]
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.level = Level.L2
    sim.level_status = DataStatus.VALID
    sim.position_status = DataStatus.VALID
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.FULL_SUPERVISION


# --- Category 9: BPMN-Specific Logic Paths (5 Tests) ---

def test_path_bpmN_checkpos_vs_checktrain_rbc(setup_test_logger):
    script = [
        "DRIVER_007", '2', True, "session_open",
        "validate_position",
        'TD', "DATA", "TRN", "ack", "Start", "ma_fs"
    ]
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.position_status = DataStatus.INVALID 
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.FULL_SUPERVISION
    assert sim.position_status == DataStatus.VALID

def test_path_l2_ack_fail_1_retry_then_success(setup_test_logger):
    script = [
        "DRIVER_007", '2', True, "session_open", "validate_position",
        'TD', "DATA", "TRN",
        "no_ack", "ack",
        "Start", "ma_fs"
    ]
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.FULL_SUPERVISION

def test_path_l2_ack_fail_9_retries_then_success(setup_test_logger):
    script = [
        "DRIVER_007", '2', True, "session_open", "validate_position",
        'TD', "DATA", "TRN",
    ]
    script.extend(["no_ack"] * 9)
    script.append("ack")
    script.extend(["Start", "ma_fs"])

    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.FULL_SUPERVISION

def test_path_l2_sm_not_available_pos_invalid(setup_test_logger):
    script = [
        "DRIVER_007", '2', True, "session_open",
        "reject_train",
        'SH'
    ]
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.SHUNTING

def test_path_l1_sm_not_available(setup_test_logger):
    script = ["DRIVER_007", '1', 'SH']
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.SHUNTING


# --- Category 10: Complex Loop Fallback Paths (5 Tests) ---

def test_path_l2_level_pos_prefilled_id_needed(setup_test_logger):
    script = [
        "DRIVER_007",
        True, "session_open",
        'TD', "DATA", "TRN", "ack", "Start", "ma_fs"
    ]
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.level = Level.L2
    sim.level_status = DataStatus.VALID
    sim.position_status = DataStatus.VALID
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.FULL_SUPERVISION
    assert sim.driver_id == "DRIVER_007"

def test_path_l1_pos_valid_all_data_valid(setup_test_logger):
    script_l1_pos_valid = [
        "DRIVER_007", '1', 'TD', "DATA", "TRN", "Start", "Ack SR"
    ]
    sim_l1 = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script_l1_pos_valid))
    sim_l1.position_status = DataStatus.VALID
    sim_l1.train_data_status = DataStatus.VALID
    sim_l1.run_start_of_mission()
    assert sim_l1.current_mode == Mode.STAFF_RESPONSIBLE
    assert sim_l1.position_status == DataStatus.VALID

def test_path_l0_pos_valid_all_data_valid(setup_test_logger):
    script = [
        "DRIVER_007", '0', 'TD', "PREFILLED_DATA", "Start", "Ack UN"
    ]
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.position_status = DataStatus.VALID
    sim.train_data_status = DataStatus.VALID
    sim.train_running_number_status = DataStatus.VALID
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.UNFITTED

def test_path_l2_all_data_prefilled_except_pos(setup_test_logger):
    script = [
        "DRIVER_007",
        True, "session_open",
        "validate_position",
        'TD', "PREFILLED_DATA",
        "ack", "Start", "ma_fs"
    ]
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.level = Level.L2
    sim.level_status = DataStatus.VALID
    sim.position_status = DataStatus.INVALID
    sim.train_data_status = DataStatus.VALID
    sim.train_running_number_status = DataStatus.VALID
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.FULL_SUPERVISION
    assert sim.position_status == DataStatus.VALID

def test_path_l2_radio_fail_loop_td_5_times_then_sh(setup_test_logger):
    script = ["DRIVER_007", '2', False]
    for i in range(5):
        script.extend(['TD', f"DATA_{i}", f"TRN_{i}"])
    script.append('SH')
    
    sim = ERTMSOnBoardSystem(logger=setup_test_logger, controller=MockController(script))
    sim.run_start_of_mission()
    assert sim.current_mode == Mode.SHUNTING