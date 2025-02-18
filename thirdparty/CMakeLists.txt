# Copyright 2025 Alişah Özcan
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

execute_process(COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/build.sh WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(GPU-NTT)