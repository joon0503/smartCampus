function sysCall_init()
    simRemoteApi.start(19999)
end

displayText_function=function(inInts,inFloats,inStrings,inBuffer)
    -- Simply display a dialog box that prints the text stored in inStrings[1]:
    if #inStrings>=1 then
        sim.displayDialog('Message from the remote API client',inStrings[1],sim.dlgstyle_ok,false)
        return {},{},{'message was displayed'},'' -- return a string
    end
end

createCuboid_function=function(inInts,inFloats,inStrings,inBuffer)
 -- Define 'constants' to improve code readability
    local CULLED_BACKFACES=1
    local VISIBLE_EDGES=2
    local APPEAR_SMOOTH=4
    local RESPONDABLE_SHAPE=8
    local STATIC_SHAPE=16
    local CYL_OPEN_ENDS=32

    local tblSize={inFloats[1], inFloats[2], inFloats[3]}
    local hndShape=sim.createPureShape(
        0, -- primitiveType
        VISIBLE_EDGES + RESPONDABLE_SHAPE + STATIC_SHAPE, --options
        tblSize, --sizes
        0, --mass
        nil --precision
        )

    if inInts[1] >= 0 then
        sim.setObjectParent( hndShape, inInts[1], true)
    end

    sim.setObjectName(hndShape,inStrings[1])
    sim.setObjectPosition(hndShape,-1,{inFloats[4], inFloats[5], inFloats[6]})       -- move to position 
    local result = sim.setObjectSpecialProperty(hndShape, sim.objectspecialproperty_detectable_all) -- make it detectable

    return {},{},{},''

    -- Create a dummy object with specific name and coordinates
 --   if #inStrings>=1 and #inFloats>=2 then
 --       local dummyHandle=sim.createDummy(0.05)
 --       local position={inInts[2],inInts[3],inInts[4]}
 --       local errorReportMode=sim.getInt32Parameter(sim.intparam_error_report_mode)
 --       sim.setInt32Parameter(sim.intparam_error_report_mode,0) -- temporarily suppress error output (because we are not allowed to have two times the same object name)
 --       sim.setObjectName(dummyHandle,inStrings[1])
 --       sim.setInt32Parameter(sim.intparam_error_report_mode,errorReportMode) -- restore the original error report mode
 --       sim.setObjectPosition(dummyHandle,-1,inFloats)
 --       return {dummyHandle},{},{},'' -- return the handle of the created dummy
 --   end
end

createDummy_function=function(inInts,inFloats,inStrings,inBuffer)
    local hndShape=sim.createDummy(
        1, -- Size
        nil --Color
        )
    sim.setObjectName(hndShape,inStrings[1])
    sim.setObjectPosition(hndShape,-1,{inFloats[1], inFloats[2], inFloats[3]})       -- move to position 
        
    if inInts[1] > 0 then
        sim.setObjectParent(hndShape, inInts[1], true)
    end


    return {hndShape},{},{},''

    -- Create a dummy object with specific name and coordinates
 --   if #inStrings>=1 and #inFloats>=2 then
 --       local dummyHandle=sim.createDummy(0.05)
 --       local position={inInts[2],inInts[3],inInts[4]}
 --       local errorReportMode=sim.getInt32Parameter(sim.intparam_error_report_mode)
 --       sim.setInt32Parameter(sim.intparam_error_report_mode,0) -- temporarily suppress error output (because we are not allowed to have two times the same object name)
 --       sim.setObjectName(dummyHandle,inStrings[1])
 --       sim.setInt32Parameter(sim.intparam_error_report_mode,errorReportMode) -- restore the original error report mode
 --       sim.setObjectPosition(dummyHandle,-1,inFloats)
 --       return {dummyHandle},{},{},'' -- return the handle of the created dummy
 --   end
end

executeCode_function=function(inInts,inFloats,inStrings,inBuffer)
    -- Execute the code stored in inStrings[1]:
    if #inStrings>=1 then
        return {},{},{loadstring(inStrings[1])()},'' -- return a string that contains the return value of the code execution
    end
end

--inInts: [#Veh, #sensor, vehicle0, sensor0 ~ 8, goalpoint0, vehicle1, sensor0~17, goalpoint1...]
-- Output
-- single array with all information 
-- [vehpos x, y, z, sensor distance0~8, goal pos x,y,z, veh1 pos x,y,z,...]
getVehicleState_function=function(inInts,inFloats,inStrings,inBuffer)
    data ={}

    -- Take veh and sensor number
    numVeh = inInts[1]
    numSensor = inInts[2]

    k=1                     -- data index counter
    handle_counter = 3
    for v=1,numVeh do       -- for each lane
        veh_pos = sim.getObjectPosition(inInts[handle_counter], -1) 
        

        -- update veh pos data
        data[k] = veh_pos[1]
        data[k+1] = veh_pos[2]
        data[k+2] = veh_pos[3]
        k=k+3

        -- veh orientation
        veh_ori = sim.getObjectOrientation(inInts[handle_counter],-1)
        data[k] = veh_ori[1]
        data[k+1] = veh_ori[2]
        data[k+2] = veh_ori[3]
        k = k+3

        handle_counter = handle_counter + 1

        -- Sensor Data
        for i=1,numSensor do
            dState, dDistance, dPoint, dObj, dVec = sim.readProximitySensor( inInts[handle_counter] )
            handle_counter = handle_counter + 1

            if dState == 0 then
                dDistance = 20.0
            end
            
            data[k] = dDistance
            k = k +1
            --Something
        end

        goal_pos = sim.getObjectPosition(inInts[handle_counter], -1) 
        handle_counter = handle_counter + 1

        -- update goal pos data
        data[k] = goal_pos[1]
        data[k+1] = goal_pos[2]
        data[k+2] = goal_pos[3]
        k = k+3
    end


    --print(data)
    return {},data,{},'' -- return a string that contains the return value of the code execution
end

--inInts:
--  inInts : [ veh1 left1, veh1 right, veh2 left, veh2 right... ]
--  inFloats: [veh1 angle, veh2 angle, ...]
-- Output
setJointPos_function=function(inInts,inFloats,inStrings,inBuffer)
    -- loop through desired angle
    for k=1, #inFloats do
        sim.setJointTargetPosition( inInts[2*k], inFloats[k] )        -- left
        sim.setJointTargetPosition( inInts[2*k-1], inFloats[k] )      -- right
    end
    return {},{},{},'' -- return a string that contains the return value of the code execution
end

