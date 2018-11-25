set(cs294_includes
    ${CMAKE_CURRENT_SOURCE_DIR}
    PARENT_SCOPE
)

# Set source files
set(
    cs294_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/cs294_interface.h
        ${CMAKE_CURRENT_SOURCE_DIR}/cs294_interface.c
        ${CMAKE_CURRENT_SOURCE_DIR}/cs294_loader.h
        ${CMAKE_CURRENT_SOURCE_DIR}/cs294_loader.c
    PARENT_SCOPE
)
